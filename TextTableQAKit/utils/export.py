#!/usr/bin/env python3
import io

import pandas as pd
import lxml.etree
import lxml.html

from tinyhtml import *
from xlsxwriter import Workbook
from .excel import write_html_table_to_excel


"""
Default methods for exporting tables to various formats.
"""
# Export methods
def export_table(
        table,
        export_format,
        cell_ids=None,
        displayed_props=None,
        include_props=True,
        linearization_style="2d",
        html_format="web",  # 'web' or 'export'
):
    if export_format == "txt":
        exported = table_to_linear(
            table,
            cell_ids=cell_ids,
            props="all" if include_props else "none",
            style=linearization_style,
            highlighted_only=False,
        )
    elif export_format == "triples":
        exported = table_to_triples(table, cell_ids=cell_ids)
    elif export_format == "html":
        exported = table_to_html(table, displayed_props, include_props, html_format)
    elif export_format == "csv":
        exported = table_to_csv(table)
    elif export_format == "xlsx":
        exported = table_to_excel(table, include_props)
    elif export_format == "json":
        exported = table_to_json(table, include_props)
    elif export_format == "reference":
        exported = get_reference(table)
    else:
        raise NotImplementedError(export_format)

    return exported

def get_reference(table):
    return table.props.get("reference")
def table_to_json(table, include_props=True):
    is_linked = table.is_linked
    data = []
    for row in table.cells:
        row_data = []
        for c in row:
            cell_data = c.serializable_props()
            if is_linked and '##[HERE STARTS THE HYPERLINKED PASSAGE]##' in cell_data["value"]:
                begin_idx = cell_data["value"].find('##[HERE STARTS THE HYPERLINKED PASSAGE]##')
                hyperlined_begin_idx = begin_idx + len('##[HERE STARTS THE HYPERLINKED PASSAGE]##')
                display_cell_value = cell_data["value"][:begin_idx]
                hyperlined_cell_value = cell_data["value"][hyperlined_begin_idx:]
                cell_data["value"] = display_cell_value
                cell_data["hyperlinked_passage"] = hyperlined_cell_value
            elif is_linked:
                cell_data["hyperlinked_passage"] = ""
            row_data.append(cell_data)
        data.append(row_data)
    j = {"data": data}

    if include_props:
        j["properties"] = table.props

    return j


def table_to_triples(table, cell_ids):
    # TODO cell ids?
    title = table.props.get("title")
    triples = []

    for i, row in enumerate(table.get_cells()):
        for j, cell in enumerate(row):
            if cell.is_header():
                continue

            row_headers = table.get_row_headers(i, j)
            col_headers = table.get_col_headers(i, j)

            if row_headers and col_headers:
                subj = row_headers[0].value
                pred = col_headers[0].value

            elif row_headers and not col_headers:
                subj = title
                pred = row_headers[0].value

            elif col_headers and not row_headers:
                subj = title
                pred = col_headers[0].value

            obj = cell.value
            triples.append([subj, pred, obj])

    return triples


def table_to_excel(table, include_props=True):
    file_stream = io.BytesIO()
    workbook = Workbook(file_stream, {"in_memory": True})
    worksheet = workbook.add_worksheet()
    write_html_table_to_excel(table, worksheet, workbook=workbook, write_table_props=include_props)
    workbook.close()

    return file_stream


def table_to_csv(table):
    df = table_to_df(table)

    # export headers only if they are not the default integer index
    export_headers = (type(df.columns) == pd.core.indexes.base.Index)

    table_csv = df.to_csv(index=False, header=export_headers)
    return table_csv


def table_to_df(table):
    table_el = _get_main_table_html(table, is_csv_output=True)
    table_html = table_el.render()
    df = pd.read_html(table_html)[0]
    return df


def table_to_html(table, displayed_props, include_props, html_format, merge = False):
    if html_format == "web" and table.props != {}:
        meta_el = _meta_to_html(table.props, displayed_props)
    elif html_format == "export" and include_props and table.props != {}:
        meta_el = _meta_to_simple_html(table.props)
    else:
        meta_el = h("div")(" ")

    table_el = _get_main_table_html(table)

    table_html = h('div')(table_el)
    meta_html = meta_el
    if not merge:
        return (lxml.etree.tostring(lxml.html.fromstring(meta_html.render()), encoding="unicode", pretty_print=True),
        lxml.etree.tostring(lxml.html.fromstring(table_html.render()), encoding="unicode", pretty_print=True))
    else:
        return lxml.etree.tostring(lxml.html.fromstring(h('div')([meta_html,table_html]).render()), encoding="unicode", pretty_print=True)

def select_props(table, props):
    if props == "none" or not table.props:
        return {}
    elif props == "factual":
        return {key: val for key, val in table.props.items() if "title" in key or "category" in key}
    elif props == "all":
        return table.props
    elif isinstance(props, list):
        return {key: table.props.get(key) for key in props}
    else:
        raise NotImplementedError(
            f"{props} properties mode is not recognized. "
            f'Available options: "none", "factual", "all", or list of keys.'
        )


def select_cells(table, highlighted_only, cell_ids):
    if cell_ids:
        return [[table.get_cell_by_id(int(idx)) for idx in cell_ids]]
    elif highlighted_only and table.has_highlights():
        return table.get_highlighted_cells()
    else:
        return table.get_cells()


def table_to_2d_str(cells, props, is_linked):

    prop_tokens = [f"{key}: {val}" for key, val in props.items()]
    prop_str = "===\n" + "\n".join(prop_tokens) + "\n===\n"

    cell_tokens = []
    for i, row in enumerate(cells):
        for j, cell in enumerate(row):
            if cell.is_dummy:
                continue
            if is_linked and '##[HERE STARTS THE HYPERLINKED PASSAGE]##' in cell.value:
                begin_idx = cell.value.find('##[HERE STARTS THE HYPERLINKED PASSAGE]##')
                display_cell_value = cell.value[:begin_idx]
            else:
                display_cell_value = cell.value
            cell_tokens.append(f"| {display_cell_value} ")
        cell_tokens.append(f"|\n")
    cell_str = "".join(cell_tokens).strip()

    return prop_str + cell_str


def table_to_linear(table_data, inclue_props):
    props = {}
    if inclue_props:
       props = table_data.props
    table = table_data.cells

    return table_to_2d_str(table, props, table_data.is_linked)



def _meta_to_html(props, displayed_props):
    meta_tbodies = []
    meta_buttons = []

    for key, value in props.items():
        meta_row_cls = "collapse show" if key in displayed_props else "collapse"
        aria_expanded = "true" if key in displayed_props else "false"

        # two wrappers around text required for collapsing
        wrapper = h("div")
        cells = [h("th")(wrapper(h("div")(key))), h("td")(wrapper(h("div")(value)))]

        meta_tbodies.append(h("tr", klass=[meta_row_cls, f"row_{key}", "collapsible"])(cells))
        meta_buttons.append(
            h(
                "button",
                type_="button",
                klass="prop-btn btn btn-fw",
                data_bs_toggle="collapse",
                data_bs_target=f".row_{key}",
                aria_expanded=aria_expanded,
                aria_controls=f"row_{key}",
                style="margin-right:8px; margin-top:5px"
            )(key)
        )


    meta_buttons_div = h("div", klass="prop-buttons")(meta_buttons)
    meta_tbody_el = h("tbody")(meta_tbodies)
    meta_table_el = h("table", klass="table table-sm caption-top meta-table")(meta_tbody_el)
    meta_el = h("div")(meta_buttons_div, meta_table_el)
    return meta_el


def _meta_to_simple_html(props):
    meta_trs = []
    for key, value in props.items():
        meta_trs.append([h("th")(key), h("td")(value)])

    meta_tbodies = [h("tr")(tds) for tds in meta_trs]
    meta_tbody_el = h("tbody")(meta_tbodies)
    meta_table_el = h("table", klass="table table-sm caption-top meta-table")(h("caption")("properties"), meta_tbody_el)
    return meta_table_el


def _get_main_table_html(table, is_csv_output = False):
    if is_csv_output:
        is_linked = False
    else:
        is_linked = table.is_linked
    trs = []
    for row in table.cells:
        tds = []
        for c in row:
            if c.is_dummy:
                continue

            eltype = "th" if c.is_header else "td"

            if is_linked and '##[HERE STARTS THE HYPERLINKED PASSAGE]##' in c.value:
                begin_idx = c.value.find('##[HERE STARTS THE HYPERLINKED PASSAGE]##')
                hyperlined_begin_idx = begin_idx+len('##[HERE STARTS THE HYPERLINKED PASSAGE]##')
                display_cell_value = c.value[:begin_idx]
                hyperlined_cell_value = c.value[hyperlined_begin_idx:]
                hyperlined_cell_value = add_dropdown_html(hyperlined_cell_value, c.idx)
                display_cell_value = [display_cell_value, hyperlined_cell_value]
            else:
                display_cell_value = c.value

            td_el = h(eltype, colspan=c.colspan, rowspan=c.rowspan, cell_idx=c.idx)(display_cell_value)

            # if c.is_highlighted:
            #     td_el.tag.attrs["class"] = "table-active"

            tds.append(td_el)
        trs.append(tds)

    tbodies = [h("tr")(tds) for tds in trs]
    tbody_el = h("tbody", id="main-table-body")(tbodies)
    table_el = h("table", klass="table table-sm no-footer table-bordered caption-top main-table",role="grid")(tbody_el)

    return table_el

def add_dropdown_html(text, idx):
    link_head = h(
        "a",
        klass="dropdown-toggle",
        id=f"messageDropdown-{idx}",
        href="#",
        data_bs_toggle="dropdown",
        aria_expanded="false"
    )(h("i", klass="mdi mdi-link mx-0")(""))
    link_body = h(
        "div",
        klass ="dropdown-menu",
        aria_labelledby=f"messageDropdown-{idx}"
    )(h("p", klass="drop-txt")(text))
    dropdowm_html = h("span")([link_head,link_body])

    return dropdowm_html







