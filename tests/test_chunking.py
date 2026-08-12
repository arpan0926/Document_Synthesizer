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


if __name__ == "__main__":
    unittest.main()
