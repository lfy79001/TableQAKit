from xlsxwriter import Workbook


COLORS = {"light_yellow": "#fffbcb", "light_gray": "#ececec", "gray": "#b7b7b7"}

STYLES = {
    "data_table": {"border": 1, "border_color": COLORS["gray"]},
    "data_table_header": {"bold": True, "border": 1, "border_color": COLORS["gray"]},
    "data_table_active": {"bg_color": COLORS["light_yellow"], "border": 1, "border_color": COLORS["gray"]},
    "data_table_header_active": {
        "bold": True,
        "bg_color": COLORS["light_yellow"],
        "border": 1,
        "border_color": COLORS["gray"],
    },
    "bold": {"bold": True},
    "ann_table_header": {"bold": True, "bg_color": COLORS["light_gray"], "border": 1, "border_color": COLORS["gray"]},
    "ann_table_delim": {
        "bold": True,
        "bg_color": COLORS["light_gray"],
        "top": 1,
        "top_color": COLORS["gray"],
        "bottom": 1,
        "bottom_color": COLORS["gray"],
    },
}


def write_merged_cells(cell, row_num, col_num, merge_cells, merged_cells, worksheet, cell_format):
    value = str(cell.value)
    end_row = row_num + cell.rowspan - 1
    end_col = col_num + cell.colspan - 1

    for r in range(row_num, end_row + 1):
        for c in range(col_num, end_col + 1):
            merged_cells.add((r, c))
            if not merge_cells:  # setting the format for all cells separately
                worksheet.write(r, c, "", cell_format)

    if merge_cells:
        worksheet.merge_range(row_num, col_num, end_row, end_col, value, cell_format)
    else:
        worksheet.write(row_num, col_num, value, cell_format)

    return merged_cells


def write_html_table_to_excel(
    table,
    worksheet,
    workbook=None,
    start_row=0,
    start_col=0,
    write_table_props=False,
):

    style_objs = {k: workbook.add_format(v) for k, v in STYLES.items()}

    if write_table_props:
        worksheet.write(start_row, start_col, "properties", style_objs["bold"])
        start_row += 1

        for prop_name in table.props.keys():
            worksheet.write(start_row, start_col, prop_name, style_objs["bold"])
            worksheet.write(start_row, start_col + 1, table.props.get(prop_name, ""))
            start_row += 1

        worksheet.write(start_row, start_col, "")
        start_row += 1
        worksheet.write(start_row, start_col, "data", style_objs["bold"])
        start_row += 1

    row_num = start_row

    for row in table.cells:
        col_num = start_col
        for cell in row:
            if cell.is_dummy:
                continue
            style_key = "data_table"
            if cell.is_col_header or cell.is_row_header:
                style_key += "_header"
            # if cell.is_highlighted:
            #     style_key += "_active"
            cell_format = style_objs[style_key]
            worksheet.write(row_num, col_num, str(cell.value), cell_format)

            col_num += cell.colspan
        row_num += 1

    return row_num


def write_annotation_to_excel(tables, prop_list, ann_columns, out_file):
    """
    Write multiple tables to excel for manual annotation.
    :param tables: List[Dict]: list of dicts, where dict is all info about the table,
                               including the table as a `Table` object
    :param prop_list: List[str]: list of properties that are selected by the user
                                 to be included in the annotation
    :param ann_columns: List[str]: list of additional annotation columns to include in the annotation table
    :param out_file: str: path for output file
    :return: None, results are written in the file
    """

    workbook = Workbook(out_file)
    worksheet = workbook.add_worksheet()
    style_objs = {k: workbook.add_format(v) for k, v in STYLES.items()}

    ann_header = ["table_id"] + ann_columns  # all annotation columns
    worksheet.write_row(0, 0, ann_header + ["property_name", "property_value", "table"])
    worksheet.set_row(0, None, style_objs["ann_table_header"])

    start_row = 1
    start_col = len(ann_header)

    for table_info in tables:
        # writing table id
        worksheet.write(start_row, 0, table_info.get("table_id", ""))

        # writing properties
        props_end_row = start_row + len(prop_list)
        for i, prop_name in enumerate(prop_list):
            worksheet.write(start_row + i, start_col, prop_name, style_objs["bold"])
            worksheet.write(start_row + i, start_col + 1, table_info.get(prop_name, ""))

        # writing table
        table_end_row = write_html_table_to_excel(
            table_info["table"],
            worksheet,
            style_objs=style_objs,
            start_row=start_row,
            start_col=start_col + 2,
            merge_cells=False,
        )

        # writing delimiter
        end_row = max(props_end_row, table_end_row)
        worksheet.set_row(end_row, None, style_objs["ann_table_delim"])
        start_row = end_row + 1

    workbook.close()
