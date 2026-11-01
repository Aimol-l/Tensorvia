#!/usr/bin/env python3
"""tools/coverage_matrix.py 的测试（stdlib unittest，无第三方依赖）。

覆盖矩阵是 FR-1 的守门人：算子 × 后端的每个格子要么有 `XxxImpl<Device::X>`
特化（native），要么在 tools/coverage_overrides.toml 里有显式标注（emulated /
unsupported），不允许"推不出来也不标注"的格子。这些测试锁住的就是这条契约。

运行：python3 -I tests/test_coverage_matrix.py
"""

import pathlib
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import tools.coverage_matrix as cm  # noqa: E402

TOOL = REPO / "tools" / "coverage_matrix.py"

# ---------------------------------------------------------------- 合成树助手

SYNTH_OPS = """\
#pragma once

namespace ops {

Tensor Sum(const Tensor& a);
Tensor Prod(const Tensor& a);
Tensor Erf(const Tensor& a);
Tensor Split(const Tensor& t, int64_t dim);
void println(Tensor& a);

struct SortResult {
    Tensor values;
    Tensor indices;
};

}  // namespace ops
"""


def header(*impls: str) -> str:
    """把 ('Sum', 'CPU') 这样的对子渲染成一份后端 ops 头。"""
    lines = ["#pragma once", "", "namespace ops {", ""]
    for name, device in impls:
        lines.append(f"template <> struct {name}Impl<Device::{device}> {{")
        lines.append("    static Tensor execute(const Tensor& a);")
        lines.append("};")
        lines.append("")
    lines.append("}  // namespace ops")
    return "\n".join(lines) + "\n"


def build_tree(root: pathlib.Path, ops_hpp: str, headers: dict, overrides: str) -> pathlib.Path:
    (root / "include" / "tensorvia").mkdir(parents=True, exist_ok=True)
    (root / "include" / "tensorvia" / "ops.hpp").write_text(ops_hpp, encoding="utf-8")
    (root / "tools").mkdir(parents=True, exist_ok=True)
    (root / "tools" / "coverage_overrides.toml").write_text(overrides, encoding="utf-8")
    for backend, text in headers.items():
        path = root / cm.IMPL_HEADER_REL[backend]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def synth_tree(root: pathlib.Path, headers=None, overrides: str = "") -> pathlib.Path:
    return build_tree(root, SYNTH_OPS, headers or {}, overrides)


def full_headers(cpu=(), cuda=("Sum",), sycl=("Sum",), vulkan=("Sum",)) -> dict:
    return {
        "CPU": header(*[("Sum", "CPU"), ("Prod", "CPU"), ("Erf", "CPU"),
                        *[(n, "CPU") for n in cpu]]),
        "CUDA": header(*[(n, "CUDA") for n in cuda]),
        "SYCL": header(*[(n, "SYCL") for n in sycl]),
        "VULKAN": header(*[(n, "VULKAN") for n in vulkan]),
    }


# 完整标注的基准 fixture：Sum 四后端 native；Prod/Erf 仅 CPU，设备端显式拒绝；
# Split 四后端组合实现（部件 Sum 到处受支持）。
BASE_OVERRIDES = """
[host_side]
names = ["println", "SortResult"]

[emulated]
Split = { note = "由 narrow 视图与 Concat 拼成", parts = ["Sum"] }

[unsupported]
Erf.devices = { note = "仅 CPU 有 ErfImpl" }
Prod.devices = { note = "仅 CPU 有 ProdImpl" }
"""


def base_tree(root: pathlib.Path, headers=None, overrides: str = BASE_OVERRIDES) -> pathlib.Path:
    return synth_tree(root, headers if headers is not None else full_headers(), overrides)


class SyntheticTreeTests(unittest.TestCase):
    """合成树：解析规则与标注契约。"""

    def test_rows_are_public_ops_minus_host_side(self):
        with tempfile.TemporaryDirectory() as tmp:
            matrix = cm.build_matrix(base_tree(pathlib.Path(tmp)))
        self.assertEqual(sorted(matrix.rows), ["Erf", "Prod", "Split", "Sum"])
        self.assertEqual(sorted(matrix.host_side), ["SortResult", "println"])

    def test_multiline_specialization_is_detected(self):
        text = ("#pragma once\nnamespace ops {\n"
                "template <>\nstruct SumImpl<Device::CUDA> {\n"
                "    static Tensor execute(const Tensor& a);\n};\n}\n")
        with tempfile.TemporaryDirectory() as tmp:
            headers = full_headers()
            headers["CUDA"] = text
            matrix = cm.build_matrix(base_tree(pathlib.Path(tmp), headers))
        self.assertEqual(matrix.rows["Sum"]["CUDA"].value, "native")

    def test_commented_out_specialization_is_ignored(self):
        text = ("#pragma once\nnamespace ops {\n"
                "// template <> struct SumImpl<Device::CUDA> {\n"
                "/* template <> struct SumImpl<Device::CUDA> { */\n"
                "}\n")
        overrides = BASE_OVERRIDES.replace(
            'Split = { note = "由 narrow 视图与 Concat 拼成", parts = ["Sum"] }',
            'Split = { note = "由 narrow 视图与 Concat 拼成" }').replace(
            'Prod.devices = { note = "仅 CPU 有 ProdImpl" }',
            'Prod.devices = { note = "仅 CPU 有 ProdImpl" }\nSum.CUDA = { note = "注释掉了" }')
        with tempfile.TemporaryDirectory() as tmp:
            headers = full_headers()
            headers["CUDA"] = text
            matrix = cm.build_matrix(base_tree(pathlib.Path(tmp), headers, overrides))
        self.assertEqual(matrix.rows["Sum"]["CUDA"].value, "unsupported")

    def test_partial_specialization_is_rejected(self):
        text = ("#pragma once\nnamespace ops {\n"
                "template <typename T> struct SumImpl<Device::CUDA> {\n"
                "    static Tensor execute(const Tensor& a);\n};\n}\n")
        with tempfile.TemporaryDirectory() as tmp:
            headers = full_headers()
            headers["CUDA"] = text
            root = base_tree(pathlib.Path(tmp), headers)
            with self.assertRaises(cm.CoverageError) as ctx:
                cm.build_matrix(root)
        message = str(ctx.exception)
        self.assertIn("partial", message)
        self.assertIn("Sum", message)
        self.assertIn("CUDA", message)

    def test_impl_name_maps_to_public_op_case_insensitively(self):
        ops = ("#pragma once\nnamespace ops {\n"
               "Tensor Argmax(const Tensor& a);\n"
               "struct SortResult { Tensor values; };\n}\n")
        overrides = ("[host_side]\nnames = [\"SortResult\"]\n"
                     "\n[unsupported]\nArgmax.devices = { note = \"仅 CPU\" }\n")
        with tempfile.TemporaryDirectory() as tmp:
            root = build_tree(pathlib.Path(tmp), ops,
                              {"CPU": header(("ArgMax", "CPU")), "CUDA": header(),
                               "SYCL": header(), "VULKAN": header()},
                              overrides)
            matrix = cm.build_matrix(root)
        self.assertEqual(matrix.rows["Argmax"]["CPU"].value, "native")

    def test_impl_without_public_op_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            headers = full_headers(cpu=["Zzz"])
            root = base_tree(pathlib.Path(tmp), headers)
            with self.assertRaises(cm.CoverageError) as ctx:
                cm.build_matrix(root)
        self.assertIn("ZzzImpl", str(ctx.exception))

    def test_unannotated_cell_errors_naming_op_and_backend(self):
        # 拿掉两条 unsupported 标注：Prod/Erf 的设备端格子一个都不许猜。
        overrides = BASE_OVERRIDES.replace(
            'Erf.devices = { note = "仅 CPU 有 ErfImpl" }\n', "").replace(
            'Prod.devices = { note = "仅 CPU 有 ProdImpl" }\n', "")
        with tempfile.TemporaryDirectory() as tmp:
            root = base_tree(pathlib.Path(tmp), None, overrides)
            with self.assertRaises(cm.CoverageError) as ctx:
                cm.build_matrix(root)
        message = str(ctx.exception)
        for op in ("Prod", "Erf"):
            self.assertIn(op, message)
        self.assertIn("CUDA", message)

    def test_override_on_native_cell_errors(self):
        overrides = BASE_OVERRIDES.replace(
            'Prod.devices = { note = "仅 CPU 有 ProdImpl" }',
            'Prod.devices = { note = "仅 CPU 有 ProdImpl" }\nSum.CUDA = { note = "标错了" }')
        with tempfile.TemporaryDirectory() as tmp:
            root = base_tree(pathlib.Path(tmp), None, overrides)
            with self.assertRaises(cm.CoverageError) as ctx:
                cm.build_matrix(root)
        message = str(ctx.exception)
        self.assertIn("Sum", message)
        self.assertIn("CUDA", message)

    def test_bare_and_per_backend_override_overlap_errors(self):
        overrides = BASE_OVERRIDES.replace(
            'Erf.devices = { note = "仅 CPU 有 ErfImpl" }',
            'Erf.devices = { note = "仅 CPU 有 ErfImpl" }\nErf.CUDA = { note = "又标一次" }')
        with tempfile.TemporaryDirectory() as tmp:
            root = base_tree(pathlib.Path(tmp), None, overrides)
            with self.assertRaises(cm.CoverageError) as ctx:
                cm.build_matrix(root)
        self.assertIn("Erf", str(ctx.exception))

    def test_emulated_parts_must_be_supported_on_same_backends(self):
        # 基准：Split 由 Sum 拼成，Sum 四后端 native → 组合声明成立。
        with tempfile.TemporaryDirectory() as tmp:
            matrix = cm.build_matrix(base_tree(pathlib.Path(tmp)))
        self.assertEqual(matrix.rows["Split"]["CUDA"].value, "emulated")

        # Sum 只剩 CPU 实现时，Split 的设备端组合声明必须报错。
        headers = full_headers(cuda=(), sycl=(), vulkan=())
        overrides = BASE_OVERRIDES + 'Sum.devices = { note = "仅 CPU" }\n'
        with tempfile.TemporaryDirectory() as tmp:
            root = base_tree(pathlib.Path(tmp), headers, overrides)
            with self.assertRaises(cm.CoverageError) as ctx:
                cm.build_matrix(root)
        message = str(ctx.exception)
        self.assertIn("Split", message)
        self.assertIn("Sum", message)

    def test_host_side_entry_must_be_a_public_op(self):
        overrides = BASE_OVERRIDES.replace(
            'names = ["println", "SortResult"]', 'names = ["println", "NoSuchOp"]')
        with tempfile.TemporaryDirectory() as tmp:
            root = base_tree(pathlib.Path(tmp), None, overrides)
            with self.assertRaises(cm.CoverageError) as ctx:
                cm.build_matrix(root)
        self.assertIn("NoSuchOp", str(ctx.exception))

    def test_core_impl_evidence_is_verified(self):
        ops = "#pragma once\nnamespace ops {\nTensor Erf(const Tensor& a);\n}\n"
        overrides = ("[core_impl]\n"
                     "Erf = { note = \"宿主侧直实现\", evidence = \"src/core/ops.cpp:2\" }\n"
                     "\n[unsupported]\nErf.devices = { note = \"require_cpu 拒绝\" }\n")
        with tempfile.TemporaryDirectory() as tmp:
            root = build_tree(pathlib.Path(tmp), ops,
                              {"CPU": "", "CUDA": header(), "SYCL": header(),
                               "VULKAN": header()},
                              overrides)
            (root / "src" / "core").mkdir(parents=True)
            (root / "src" / "core" / "ops.cpp").write_text(
                "line1\n    require_cpu(a, \"erf\");\n", encoding="utf-8")
            matrix = cm.build_matrix(root)
        self.assertEqual(matrix.rows["Erf"]["CPU"].value, "native")

        with tempfile.TemporaryDirectory() as tmp:
            root = build_tree(pathlib.Path(tmp), ops,
                              {"CPU": "", "CUDA": header(), "SYCL": header(),
                               "VULKAN": header()},
                              overrides)
            (root / "src" / "core").mkdir(parents=True)
            (root / "src" / "core" / "ops.cpp").write_text(
                "line1\nnothing about the op here\n", encoding="utf-8")
            with self.assertRaises(cm.CoverageError) as ctx:
                cm.build_matrix(root)
        self.assertIn("Erf", str(ctx.exception))

    def test_render_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = cm.render(cm.build_matrix(base_tree(pathlib.Path(tmp))))
            second = cm.render(cm.build_matrix(base_tree(pathlib.Path(tmp))))
        self.assertEqual(first, second)


class RealTreeTests(unittest.TestCase):
    """真实仓库：矩阵完整性与 CLI 行为。"""

    def copy_repo(self, root: pathlib.Path) -> None:
        for rel in (cm.OPS_HPP_REL, cm.OVERRIDES_REL, "src/core/ops.cpp"):
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text((REPO / rel).read_text(encoding="utf-8"), encoding="utf-8")
        for rel in cm.IMPL_HEADER_REL.values():
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text((REPO / rel).read_text(encoding="utf-8"), encoding="utf-8")

    def test_every_public_op_is_a_row_or_host_side(self):
        public = set(cm.parse_public_ops(REPO / cm.OPS_HPP_REL))
        matrix = cm.build_matrix(REPO)
        rows = set(matrix.rows)
        host = set(matrix.host_side)
        self.assertEqual(public, rows | host, "公开算子必须全部有归属：矩阵行或宿主侧")
        self.assertFalse(rows & host)

    def test_every_row_has_four_cells(self):
        matrix = cm.build_matrix(REPO)
        for op, cells in matrix.rows.items():
            self.assertEqual(sorted(cells), ["CPU", "CUDA", "SYCL", "VULKAN"], op)

    def test_stats_counts_match_rows(self):
        matrix = cm.build_matrix(REPO)
        stats = matrix.stats()
        for backend in cm.BACKENDS:
            counts = stats[backend]
            total = counts["native"] + counts["emulated"] + counts["unsupported"]
            self.assertEqual(total, len(matrix.rows), backend)
            self.assertEqual(counts["supported"], counts["native"] + counts["emulated"])

    def test_cpu_only_ops_are_unsupported_on_devices(self):
        # FR-2 点名的六个算子在设备端必须显式拒绝（矩阵是唯一事实源）。
        matrix = cm.build_matrix(REPO)
        for op in ("Erf", "MaskedSelect", "MaskedFill", "Sort", "Topk", "Prod"):
            for backend in cm.DEVICE_BACKENDS:
                self.assertEqual(matrix.rows[op][backend].value, "unsupported",
                                 f"{op} on {backend}")

    def test_check_passes_on_clean_tree(self):
        result = subprocess.run(
            [sys.executable, "-I", str(TOOL), "--check"],
            cwd=REPO, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_write_then_check_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self.copy_repo(root)
            written = subprocess.run(
                [sys.executable, "-I", str(TOOL), "--write", "--root", str(root)],
                cwd=REPO, capture_output=True, text=True)
            self.assertEqual(written.returncode, 0, written.stderr)
            self.assertTrue((root / "docs" / "coverage.md").is_file())
            checked = subprocess.run(
                [sys.executable, "-I", str(TOOL), "--check", "--root", str(root)],
                cwd=REPO, capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_check_fails_when_vulkan_specialization_removed(self):
        # 计划 Step 3：删掉 vulkan_ops.hpp 里一个特化，--check 必须失败并指名算子与后端。
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self.copy_repo(root)
            vulkan_rel = cm.IMPL_HEADER_REL["VULKAN"]
            vulkan_path = root / vulkan_rel
            vulkan_text = vulkan_path.read_text(encoding="utf-8")
            victim = "template <> struct ReluImpl<Device::VULKAN> {"
            self.assertIn(victim, vulkan_text)
            vulkan_path.write_text(vulkan_text.replace(victim + "\n", "", 1),
                                   encoding="utf-8")
            checked = subprocess.run(
                [sys.executable, "-I", str(TOOL), "--check", "--root", str(root)],
                cwd=REPO, capture_output=True, text=True)
        self.assertEqual(checked.returncode, 1)
        self.assertIn("Relu", checked.stderr)
        self.assertIn("VULKAN", checked.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
