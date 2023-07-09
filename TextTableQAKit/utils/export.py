#!/usr/bin/env python3
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
    j = {"data": [[c.serializable_props() for c in row] for row in table.get_cells()]}

    if include_props and table.props is not None:
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
    workbook = Workbook("tmp.xlsx", {"in_memory": True})
    worksheet = workbook.add_worksheet()
    write_html_table_to_excel(table, worksheet, workbook=workbook, write_table_props=include_props)

    return workbook


def table_to_csv(table):
    df = table_to_df(table)

    # export headers only if they are not the default integer index
    export_headers = (type(df.columns) == pd.core.indexes.base.Index)

    table_csv = df.to_csv(index=False, header=export_headers)
    return table_csv


def table_to_df(table):
    table_el = _get_main_table_html(table)
    table_html = table_el.render()
    df = pd.read_html(table_html)[0]
    return df


def table_to_html(table, displayed_props, include_props, html_format):
    if html_format == "web" and table.props is not None:
        meta_el = _meta_to_html(table.props, displayed_props)
    elif html_format == "export" and include_props and table.props is not None:
        meta_el = _meta_to_simple_html(table.props)
    else:
        meta_el = None

    table_el = _get_main_table_html(table)

    table_html = h('div')(table_el).render()
    meta_html = meta_el.render()
    return (lxml.etree.tostring(lxml.html.fromstring(meta_html), encoding="unicode", pretty_print=True),
    lxml.etree.tostring(lxml.html.fromstring(table_html), encoding="unicode", pretty_print=True))



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


def table_to_2d_str(cells, props):
    prop_tokens = [f"{key}: {val}" for key, val in props.items()]
    prop_str = "===\n" + "\n".join(prop_tokens) + "\n===\n"

    cell_tokens = []
    for i, row in enumerate(cells):
        for j, cell in enumerate(row):
            if cell.is_dummy:
                continue

            cell_tokens.append(f"| {cell.value} ")
        cell_tokens.append(f"|\n")
    cell_str = "".join(cell_tokens).strip()

    return prop_str + cell_str


def table_to_markers_str(cells, props):
    tokens = [f"[P] {key}: {val}" for key, val in props.items()]

    for i, row in enumerate(cells):
        tokens.append("[R]")

        for j, cell in enumerate(row):
            if cell.is_dummy:
                continue

            tokens.append("[H]" if cell.is_header else "[C]")
            tokens.append(cell.value)

    return " ".join(tokens)


def table_to_indexed_str(cells, props):
    tokens = [f"[P] {key}: {val}" for key, val in props.items()]
    for i, row in enumerate(cells):
        for j, cell in enumerate(row):
            if cell.is_dummy:
                continue

            tokens.append(f"[{i}][{j}]")
            tokens.append(cell.value)
    return " ".join(tokens)


def table_to_linear(
    table,
    cell_ids=None,
    props="factual",  # 'all', 'factual', 'none', or list of keys
    style="2d",  # 'index', 'markers', '2d'
    highlighted_only=False,
):
    props_to_include = select_props(table, props)
    cells_to_include = select_cells(table, highlighted_only, cell_ids)

    if style == "2d":
        return table_to_2d_str(cells_to_include, props_to_include)
    elif style == "markers":
        return table_to_markers_str(cells_to_include, props_to_include)
    elif style == "index":
        return table_to_indexed_str(cells_to_include, props_to_include)
    else:
        raise NotImplementedError(
            f"{style} linearization style is not recognized. " f'Available options: "index", "markers", or "2d".'
        )


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


def _get_main_table_html(table):

    trs = []
    for row in table.cells:
        tds = []
        for c in row:
            if c.is_dummy:
                continue

            eltype = "th" if c.is_header else "td"
            td_el = h(eltype, colspan=c.colspan, rowspan=c.rowspan, cell_idx=c.idx)(c.value)

            if c.is_highlighted:
                td_el.tag.attrs["class"] = "table-active"

            tds.append(td_el)
        trs.append(tds)

    tbodies = [h("tr")(tds) for tds in trs]
    tbody_el = h("tbody", id="main-table-body")(tbodies)
    table_el = h("table", klass="table dataTable table-sm no-footer table-bordered caption-top main-table",role="grid")(h("caption")("data"), tbody_el)

    return table_el




