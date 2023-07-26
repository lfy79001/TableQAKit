import copy
import logging
import datasets


logger = logging.getLogger(__name__)


class Cell:
    """
    Table cell
    """

    def __init__(
        self,
        value=None,
        idx=None,
        colspan=1,
        rowspan=1,
        is_highlighted=False,
        is_col_header=False,
        is_row_header=False,
        is_dummy=False,
        main_cell=None,
    ):
        self.idx = idx
        self.value = value
        self.colspan = colspan
        self.rowspan = rowspan
        self.is_highlighted = is_highlighted
        self.is_col_header = is_col_header
        self.is_row_header = is_row_header
        self.is_dummy = is_dummy
        self.main_cell = main_cell

    @property
    def is_header(self):
        return self.is_col_header or self.is_row_header

    def serializable_props(self):

        ser_props = {}
        ser_props["idx"] = self.idx
        ser_props["value"] = str(self.value)
        ser_props["is_row_header"] = self.is_row_header
        ser_props["is_col_header"] = self.is_col_header
        return ser_props

    def __repr__(self):
        return str(vars(self))


class Table:
    """
    Table object
    """

    def __init__(self):
        self.props = {}
        self.cells = []
        self.outputs = {}
        self.url = None
        self.cell_idx = 0
        self.current_row = []
        self.cell_by_ids = {}
        self.default_question = ""
        self.type = "default" # default or custom
        self.custom_table_name = "" # for custom table
        self.txt_info = []
        self.pic_info = []
        self.is_linked = False
        self.html = "" # for multihiertt dataset

    def has_highlights(self):
        return any(cell.is_highlighted for row in self.cells for cell in row)

    def save_row(self):
        if self.current_row:
            self.cells.append(self.current_row)
            self.current_row = []

    def add_cell(self, cell):
        cell.idx = self.cell_idx
        self.current_row.append(cell)
        self.cell_by_ids[self.cell_idx] = cell
        self.cell_idx += 1

    def set_cell(self, i, j, c):
        self.cells[i][j] = c

    def get_cell(self, i, j):
        try:
            return self.cells[i][j]
        except:
            return None

    def get_cell_by_id(self, idx):
        return self.cell_by_ids[idx]

    def get_flat_cells(self, highlighted_only=False):
        return [x for row in self.cells for x in row if (x.is_highlighted or not highlighted_only)]

    def get_highlighted_cells(self):
        return self.get_cells(highlighted_only=True)

    def get_cells(self, highlighted_only=False):
        if highlighted_only:
            cells = []
            for row in self.cells:
                row_cells = [c for c in row if c.is_highlighted]
                if row_cells:
                    cells.append(row_cells)
            return cells
        else:
            return self.cells

    def get_row_headers(self, row_idx, column_idx):
        try:
            headers = [c for c in self.get_cells()[row_idx][:column_idx] if c.is_row_header]
            return headers

        except Exception as e:
            logger.exception(e)

    def get_col_headers(self, row_idx, column_idx):
        try:
            headers = []
            for i, row in enumerate(self.get_cells()):
                if i == row_idx:
                    return headers

                if len(row) > column_idx and row[column_idx].is_col_header:
                    headers.append(row[column_idx])

            return headers

        except Exception as e:
            logger.exception(e)

    def __repr__(self):
        return str(self.__dict__)



class TabularDataset:
    """
    Base class for the datasets
    """

    def __init__(self, path):
        self.splits = ["train", "dev", "test"]
        self.data = {split: [] for split in self.splits}
        self.tables = {split: {} for split in self.splits}
        self.path = path
        self.dataset_info = {}
        self.name = None

    def load(self, split):
        """
        Load the dataset. Path can be specified for loading from a directory
        or omitted if the dataset is loaded from HF.
        """
        raise NotImplementedError



    def get_data(self, split, table_idx):
        return self.data[split][table_idx]

    def set_table(self, split, table_idx, table):
        self.tables[split][table_idx] = table

    def get_example_count(self, split):
        return len(self.data[split])

    def has_split(self, split):
        return bool(self.data[split])

    def get_table(self, split, table_idx, edited_cells=None):
        table = self.tables[split].get(table_idx)
        if edited_cells:
            table_modif = copy.deepcopy(table)
            for cell_id, val in edited_cells.items():
                cell = table_modif.get_cell_by_id(int(cell_id))
                cell.value = val
            table = table_modif
        return table

    def prepare_table(self, entry):
        return NotImplementedError

    def get_info(self):
        return self.dataset_info

class HFTabularDataset(TabularDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, path=None, **kwargs)

        self.split_mapping = {"train": "train", "dev": "validation", "test": "test"}
        self.dataset_info = {}


    def _load_split(self, split):
        return NotImplementedError

    def load(self, split=None):
        if split is None:
            for split in self.split_mapping.keys():
                self._load_split(split)
        else:
            self._load_split(split)

    def save_to_disk(self, split, filepath):
        self.data[split].save_to_disk(filepath)
        logger.info(f"File {filepath} saved successfully")

    def load_from_disk(self, split, filepath):
        self.data[split] = datasets.load_dataset(filepath)
