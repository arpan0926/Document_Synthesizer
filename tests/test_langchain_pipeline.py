import unittest

from langchain_core.documents import Document

from generation import build_answer_prompt
from retrieval import to_langchain_documents


class LangChainPipelineTests(unittest.TestCase):
    def test_to_langchain_documents_preserves_metadata(self) -> None:
        chunks = [
            {
                "content": "This is a test chunk.",
                "metadata": {"source_doc": "paper.pdf", "page_number": 3, "chunk_type": "text"},
            }
        ]

        documents = to_langchain_documents(chunks)

        self.assertEqual(len(documents), 1)
        self.assertIsInstance(documents[0], Document)
        self.assertEqual(documents[0].page_content, "This is a test chunk.")
        self.assertEqual(documents[0].metadata["source_doc"], "paper.pdf")
        self.assertEqual(documents[0].metadata["page_number"], 3)

    def test_build_answer_prompt_formats_question_and_context(self) -> None:
        chunks = [
            {
                "content": "Relevant context from the PDF.",
                "metadata": {"source_doc": "paper.pdf", "page_number": 5, "chunk_type": "text"},
            }
        ]

        prompt = build_answer_prompt("What is the summary?", chunks)
        rendered = prompt.format(question="What is the summary?", context="Relevant context from the PDF.")

        self.assertIn("What is the summary?", rendered)
        self.assertIn("Relevant context from the PDF.", rendered)


if __name__ == "__main__":
    unittest.main()
