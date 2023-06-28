import copy
import logging


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
        self.main_cell = main_cell  # for dummy cells

    @property
    def is_header(self):
        return self.is_col_header or self.is_row_header

    def serializable_props(self):
        props = copy.deepcopy(vars(self))
        props["value"] = str(props["value"])
        return props

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

    def get_generated_output(self, key):
        return self.outputs.get(key)

    def set_generated_output(self, key, value):
        self.outputs[key] = value

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

    def load(self, split, max_examples=None):
        """
        Load the dataset. Path can be specified for loading from a directory
        or omitted if the dataset is loaded from HF.
        """
        raise NotImplementedError

    @staticmethod
    def get_reference(table):
        return table.props.get("reference")

    def get_example_count(self, split):
        return len(self.data[split])

    def has_split(self, split):
        return bool(self.data[split])

    def get_table(self, split, table_idx, edited_cells=None):
        table = self.tables[split].get(table_idx)

        if not table:
            entry = self.data[split][table_idx]
            table = self.prepare_table(entry)
            self.tables[split][table_idx] = table

        if edited_cells:
            # make a temporary copy of the table with edited cells
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

    def export_table(
        self,
        table,
        export_format,
        cell_ids=None,
        displayed_props=None,
        include_props=True,
        linearization_style="2d",
        html_format="web",  # 'web' or 'export'
    ):
        if export_format == "txt":
            exported = self.table_to_linear(
                table,
                cell_ids=cell_ids,
                props="all" if include_props else "none",
                style=linearization_style,
                highlighted_only=False,
            )
        elif export_format == "triples":
            exported = self.table_to_triples(table, cell_ids=cell_ids)
        elif export_format == "html":
            exported = self.table_to_html(table, displayed_props, include_props, html_format)
        elif export_format == "csv":
            exported = self.table_to_csv(table)
        elif export_format == "xlsx":
            exported = self.table_to_excel(table, include_props)
        elif export_format == "json":
            exported = self.table_to_json(table, include_props)
        elif export_format == "reference":
            exported = self.get_reference(table)
        else:
            raise NotImplementedError(export_format)

        return exported

    def export(self, split, table_cfg):
        exported = []

        for i in range(self.get_example_count(split)):
            obj = {}
            for key, export_format in table_cfg["fields"].items():
                table = self.get_table(split, i)
                obj[key] = self.export_table(table, export_format=export_format)
            exported.append(obj)

        return exported

    def get_hf_dataset(
        self,
        split,
        tokenizer,
        linearize_fn=None,
        linearize_params=None,
        highlighted_only=True,
        max_length=512,
        num_proc=8,
    ):
        # linearize tables and convert to input_ids
        if linearize_params is None:
            linearize_params = {}

        if linearize_fn is None:
            linearize_fn = self.table_to_linear
            linearize_params["style"] = "markers"
            linearize_params["highlighted_only"] = highlighted_only

        def process_example(example):
            table_obj = self.prepare_table(example)
            linearized = linearize_fn(table_obj, **linearize_params)
            ref = self.get_reference(table_obj)

            tokens = tokenizer(linearized, max_length=max_length, truncation=True)
            ref_tokens = tokenizer(text_target=ref, max_length=max_length, truncation=True)
            tokens["labels"] = ref_tokens["input_ids"]

            return tokens

        logger.info(f"[tabgenie] linearizing tables using {linearize_fn}")
        lin_example = linearize_fn(self.prepare_table(self.data[split][0]), **linearize_params)
        logger.info(f"[tabgenie] linearized example ({split}/0): {lin_example}")

        processed_dataset = self.data[split].map(process_example, batched=False, num_proc=1)
        extra_columns = [
            col for col in processed_dataset.features.keys() if col not in ["labels", "input_ids", "attention_mask"]
        ]
        processed_dataset = processed_dataset.remove_columns(extra_columns)
        processed_dataset.set_format(type="torch")

        return processed_dataset

    def get_linearized_pairs(self, split, linearize_fn=None):
        if linearize_fn is None:
            linearize_fn = self.table_to_linear

        data = []
        for i, entry in enumerate(self.data[split]):
            ex = [
                linearize_fn(self.prepare_table(entry)),
                self.get_reference(self.get_table(split, i)),
            ]
            data.append(ex)

        return data

    def get_task_definition(self):
        # TODO implement for individual datasets
        return "Describe the following structured data in one sentence."

    def get_positive_examples(self):
        # TODO implement for individual datasets
        # TODO fix - split may not be loaded
        table_ex_1 = self.get_table("dev", 0)
        table_ex_2 = self.get_table("dev", 1)

        return [
            {
                "in": self.table_to_linear(table_ex_1),
                "out": self.get_reference(table_ex_1),
            },
            {
                "in": self.table_to_linear(table_ex_2),
                "out": self.get_reference(table_ex_2),
            },
        ]

    # Export methods
    def table_to_csv(self, table):
        return export.table_to_csv(table)

    def table_to_df(self, table):
        return export.table_to_df(table)

    def table_to_html(self, table, displayed_props, include_props, html_format):
        return export.table_to_html(
            table, displayed_props=displayed_props, include_props=include_props, html_format=html_format
        )

    def table_to_json(self, table, include_props=True):
        return export.table_to_json(table, include_props=include_props)

    def table_to_excel(self, table, include_props=True):
        return export.table_to_excel(table, include_props=include_props)

    def table_to_linear(
        self,
        table,
        cell_ids=None,
        props="factual",  # 'all', 'factual', 'none', or list of keys
        style="2d",  # 'index', 'markers', '2d'
        highlighted_only=False,
    ):
        return export.table_to_linear(
            table,
            cell_ids=cell_ids,
            props=props,
            style=style,
            highlighted_only=highlighted_only,
        )

    # End export methods


class HFTabularDataset(TabularDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, path=None, **kwargs)
        self.hf_id = None  # needs to be set
        self.hf_extra_config = None
        self.split_mapping = {"train": "train", "dev": "validation", "test": "test"}
        self.dataset_info = {}
        self.extra_info = {}

    def _load_split(self, split):
        hf_split = self.split_mapping[split]

        logger.info(f"Loading {self.hf_id}/{split}")
        dataset = datasets.load_dataset(
            self.hf_id,
            name=self.hf_extra_config,
            split=hf_split,
            # num_proc=4,
        )
        self.dataset_info = dataset.info.__dict__
        self.data[split] = dataset

    def load(self, split=None, max_examples=None):
        if max_examples is not None:
            logger.warning("The `max_examples` parameter is not currently supported for HF datasets")

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

    def get_info(self):
        info = {key: self.dataset_info.get(key) for key in ["citation", "description", "version", "license"]}
        info["examples"] = {}
        info["links"] = {}

        for split_name, split_info in self.dataset_info.get("splits").items():
            if split_name.startswith("val"):
                split_name = "dev"

            if split_name not in ["train", "dev", "test"]:
                continue

            info["examples"][split_name] = split_info.num_examples

        if info["version"] is not None:
            info["version"] = str(info["version"])

        if self.dataset_info.get("homepage"):
            info["links"]["homepage"] = self.dataset_info["homepage"]
        elif self.extra_info.get("homepage"):
            info["links"]["homepage"] = self.extra_info["homepage"]

        info["links"]["source"] = "https://huggingface.co/datasets/" + self.hf_id
        info["name"] = self.name

        # some info may not be present on HF, set it manually
        info.update(self.extra_info)

        return info

