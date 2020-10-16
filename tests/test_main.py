from unittest import TestCase

from main import create_copy_and_apply


class Test(TestCase):
    def test_create_copy_and_apply(self):
        copy_fn = create_copy_and_apply(lambda x: x)
        copy_fn(src="tests/test_data/test-clean/61/70968/61-70968-0000.flac",
                dst="tests/test_data/test-clean-copy/61/70968/61-70968-0000.flac")
