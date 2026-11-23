import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from colab.tools.build_notebooks import build  # noqa: E402

SOURCE = """# %% [markdown]
# # Tiêu đề
# Mô tả bước.

# %%
x = 1

# %%
!pip -q install datasets
print(x)
"""


class BuildNotebookTest(unittest.TestCase):
    def test_build_three_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "nb.py", Path(tmp) / "nb.ipynb"
            src.write_text(SOURCE, encoding="utf-8")
            build(src, dst)
            notebook = json.loads(dst.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual([cell["cell_type"] for cell in notebook["cells"]], ["markdown", "code", "code"])
        self.assertEqual("".join(notebook["cells"][0]["source"]), "# Tiêu đề\nMô tả bước.")
        self.assertEqual("".join(notebook["cells"][2]["source"]), "!pip -q install datasets\nprint(x)")
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")


if __name__ == "__main__":
    unittest.main()
