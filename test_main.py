import unittest
from unittest.mock import patch

import pandas as pd

from main import create_parser, main


class TestMain(unittest.TestCase):
    def setUp(self):
        parser = create_parser()
        self.parser = parser
        self.smalldataset = pd.read_csv("boston2017-2018.csv").sample(100)

    @patch("pandas.read_csv")
    def test_train(self, mock):
        mock.return_value = self.smalldataset

        args = self.parser.parse_args(["--do_train"])
        main(args)

    def test_predict(self):

        args = self.parser.parse_args(["--elapsed_time", "0:26:00", 
                                       "--elapsed_time_what_if", "0:25:00"])
        main(args)
