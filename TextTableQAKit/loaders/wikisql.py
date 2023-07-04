import datasets
import logging

from ..structs.data import Cell, Table, HFTabularDataset
from google.cloud import storage

logger = logging.getLogger(__name__)

class WikiSQL(HFTabularDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapping = {}
        self.hf_id = "wikisql"
        self.name = "WikiSQL"
        self.extra_info = {"license": "BSD 3-Clause"}

    @staticmethod
    def _get_title(table):
        keys_to_try = ["caption", "section_title", "page_title"]
        for key in keys_to_try:
            value = table.get(key, "").strip()
            if value:
                return value

    def _load_split(self, split):
        hf_split = self.split_mapping[split]

        logger.info(f"Loading {self.hf_id} - {split}")
        '''
        loading dataset
        '''
        dataset = datasets.load_dataset(
            self.hf_id,
            name=self.hf_extra_config,
            split=hf_split,
            # num_proc=4,
        )

        self.dataset_info = dataset.info.__dict__
        self.data[split] = dataset

    def prepare_table(self, entry):
        t = Table()
        t.type = "default"
        ######################################
        #需要补充default_QUESTION
        ####################################
        t.default_question = ""
        title = self._get_title(entry["table"])
        if title is not None:
            t.props["title"] = title

        t.props["sql"] = entry["sql"]["human_readable"]
        t.props["reference"] = entry["question"]
        t.props["id"] = entry["table"]["id"]
        t.props["name"] = entry["table"]["name"]

        for header_cell in entry["table"]["header"]:
            c = Cell()
            c.value = header_cell
            c.is_col_header = True
            t.add_cell(c)
        t.save_row()

        for row in entry["table"]["rows"]:
            for cell in row:
                c = Cell()
                c.value = cell
                t.add_cell(c)
            t.save_row()

        return t