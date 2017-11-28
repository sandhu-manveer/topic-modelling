import csv
import os

import pandas as pd
import numpy as np

class RawToCorpusConverter():

    def __init__(self, filepath):
        self.df = pd.read_csv(filepath, encoding='iso-8859-1', error_bad_lines=False)

    def getCategories(self, categoryField="_type"):
        return self.df[categoryField].unique()

    def setUpCorpus(self, preprocessedDir, categories, textField="Text", altField="notes", categoryField="_type"):
        for category in categories:
            if not os.path.exists(os.path.join(preprocessedDir, category)):
                os.mkdir(os.path.join(preprocessedDir, category))
        
        for index, row in self.df.iterrows():
            fp = open(os.path.join(preprocessedDir, row[categoryField], str(index) + '.txt'), 'w', encoding='iso-8859-1')
            if isinstance(row[textField], str):
                fp.write(row[textField])
            elif isinstance(row[textField], float):
                fp.write("")
            else:
                pass
            fp.close()

if __name__ == '__main__':
    converter = RawToCorpusConverter('path\\to\\raw_data\\items_bcgspider_4.csv')
    categories = converter.getCategories()
    converter.setUpCorpus('path\\to\\raw_corpus\\basic_corpus', categories)