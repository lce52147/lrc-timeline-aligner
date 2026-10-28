import unittest

from numeral_reading import number_to_kanji, numerals_for_alignment


class NumeralReadingTests(unittest.TestCase):
    def test_single_digit_in_japanese_line(self):
        self.assertEqual(numerals_for_alignment("1秒前の僕はもういない"), "一秒前の僕はもういない")

    def test_fullwidth_digits_and_compound_numbers(self):
        self.assertEqual(numerals_for_alignment("２人で10回"), "二人で十回")
        self.assertEqual(number_to_kanji("105"), "百五")
        self.assertEqual(number_to_kanji("2026"), "二千二十六")
        self.assertEqual(number_to_kanji("0"), "〇")

    def test_long_or_padded_runs_are_read_digit_by_digit(self):
        self.assertEqual(number_to_kanji("007"), "〇〇七")
        self.assertEqual(number_to_kanji("19990"), "一九九九〇")

    def test_english_and_digitless_lines_are_untouched(self):
        self.assertEqual(numerals_for_alignment("Take me to 1 place"), "Take me to 1 place")
        self.assertEqual(numerals_for_alignment("止まらないよね砂時計"), "止まらないよね砂時計")

    def test_decimals_are_left_alone(self):
        self.assertEqual(numerals_for_alignment("3.5倍"), "3.5倍")


if __name__ == "__main__":
    unittest.main()
