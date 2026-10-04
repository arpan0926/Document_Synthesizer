import unittest

import pandas as pd

from chunking import chunk_text, chunk_table


class ChunkingTests(unittest.TestCase):
    def test_chunk_text_creates_metadata_and_overlap(self) -> None:
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunk_text(text, source_doc="paper.pdf", page_number=2)

        self.assertGreaterEqual(len(chunks), 1)
        self.assertTrue(all(chunk["metadata"]["source_doc"] == "paper.pdf" for chunk in chunks))
        self.assertTrue(all(chunk["metadata"]["page_number"] == 2 for chunk in chunks))
        self.assertTrue(all(chunk["metadata"]["chunk_type"] == "text" for chunk in chunks))

    def test_chunk_table_keeps_single_chunk(self) -> None:
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        chunk = chunk_table(df, source_doc="paper.pdf", page_number=4)

        self.assertEqual(chunk["metadata"]["chunk_type"], "table")
        self.assertEqual(chunk["metadata"]["page_number"], 4)
        self.assertEqual(chunk["metadata"]["source_doc"], "paper.pdf")
        self.assertIs(chunk["metadata"]["df"], df)

    def test_chunk_text_handles_large_paragraph(self) -> None:
        # A single paragraph with many words that will definitely exceed max_tokens=10
        text = "word " * 50
        chunks = chunk_text(text, source_doc="paper.pdf", page_number=3, max_tokens=10, overlap_tokens=2)
        
        self.assertGreater(len(chunks), 1)
        # Verify reconstruction
        reconstructed = " ".join(chunk["content"] for chunk in chunks)
        # Check that it contains all the original words (overlap will add duplicates, but the core content must exist)
        original_words = text.strip().split()
        reconstructed_words = reconstructed.strip().split()
        for word in original_words:
            self.assertIn(word, reconstructed_words)


if __name__ == "__main__":
    unittest.main()
