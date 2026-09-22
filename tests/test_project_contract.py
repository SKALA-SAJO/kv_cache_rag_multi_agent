"""교수 재현성 점검을 위한 최소 프로젝트 구조·문서 계약 테스트."""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProjectContractTest(unittest.TestCase):
    def test_required_project_files_exist(self) -> None:
        required = {
            "app.py",
            "README.md",
            ".env.example",
            "pyproject.toml",
            "agents/base.py",
            "graph/state.py",
            "graph/workflow.py",
            "rag/ingest.py",
            "rag/retriever.py",
            "rag/external_search.py",
        }
        missing = [path for path in required if not (ROOT / path).exists()]
        self.assertEqual(missing, [])

    def test_readme_has_required_sections(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for section in [
            "Subject",
            "Overview",
            "Selected Technologies",
            "Features",
            "Tech Stack",
            "Agents",
            "Architecture",
            "Directory Structure",
            "Usage",
            "Contributors",
        ]:
            self.assertIn(section, readme)


if __name__ == "__main__":
    unittest.main()
