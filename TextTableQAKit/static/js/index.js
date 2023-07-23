var split = 'dev';
var dataset = 'finqa';
var default_model = 'T5-small';
var model = 'T5-small';
var table_idx = 0;
var total_examples = 1;
var default_result = {};
var default_question = '';
var properties_html = '';
var table_html = '';
var pictures = {};
var text = {};
var host = '210.75.240.136';
var port = '18889';
var url = '';

function mod(n, m) {
    return ((n % m) + m) % m;
}

function splitChanged() {
    split = $('#split-select').val();
    table_idx = 0;

}

function datasetChanged() {
    dataset = $('#dataset-select').val();
    table_idx = 0;
}

function defaultModelChanged() {
    default_model = $('#default-model-select').val();
    if (default_model in default_result) {
        $('#default-model-answer').html(default_model[default_model]);
    }
}

function changeModel() {
    model = $('#model-select').val();
}

function changeDefaultQuestionHtml(question) {
    $('#default-question').html(question);
}

function changeTableHtml(table) {
    $('#table-container').html(table);
}

function changePictureHtml() {
    let pictures_html = ''
    for (let pic in pictures) {
        pictures_html += '<div class="col-sm-6 col-md-4 col-lg-3"><img src="../static/img/' + pic + '" style="max-width: 100%; height: auto;"></div>'
    }
    $('#image-container').html(pictures_html);
}

function changePropertiesHtml() {
    $('#properties-container').html(properties_html);
}

function changeTextHtml() {
    $('#default-question').html(default_question);
}

function changeTotalexamples() {
    $('total-examples').html(total_examples - 1);
}

function getData() {
    $.get('http://' + host + ':' + port + '/table/default', { 'dataset_name': dataset, 'split': split, 'table_idx': table_idx },
        (data, status) => {
            log.console(status);
            log.console(data);
            if (status != 'success' || data.success == false) {
                alert('fetch table error');
                return false;
            } else {
                total_examples = data.table_cnt;
                default_result = data.generated_results;
                dataset_info = data.dataset_info;
                default_question = data.table_question;
                properties_html = data.properties_html;
                table_html = data.table_html;
                pictures = data.pictures;
                text = data.text;
                return true;
            }
        })
}

function generateAnswer() {
    // TODO pipeline
}

function nextbtn() {
    gotopage(table_idx + 1);
}

function prevbtn() {
    gotopage(table_idx - 1);
}

function startbtn() {
    gotopage(0);
}

function endbtn() {
    gotopage(total_examples - 1);
}

function gotobtn() {
    var n = $("#page-input").val();
    gotopage(n);
}

function gotopage(page) {
    table_idx = page;
    table_idx = mod(table_idx, total_examples);

    fetch_table(dataset, split, table_idx);
    $("#page-input").val(table_idx);
}

function downloadTable() {
    $.get(url + '/default/download', {
        'format': $('download-format-select').val(),
        'include_props': $('includes-properties').val(),
        'dataset_name': dataset,
        'splite': split,
        'table_idx': table_idx
    }, (data, status) => {
        console.log(status);
        if (status === 'success') {
            var downloadUrl = window.URL.createObjectURL(data);
            window.location.href = downloadUrl;
        }
    });
}

function init_page() {
    table_cnt = 0;
    total_examples = 1;
    if (getData()) {
        changeDefaultQuestionHtml();
        changeTableHtml();
        changePropertiesHtml();
        changeTextHtml();
        changeTotalexamples();
    }
}

$(document).ready(() => {
    url = "http://" + host + ':' + port;
    $('#dataset-select').val(dataset);
    $('#splite-select').val(split);
    $('#default-model-select').val(default_model);
    $('#model-select').val(model);
    $("#dataset-select").change(datasetChanged);
    $("#split-select").change(splitChanged);
    $('#default-model-select').change(defaultModelChanged);
    $('#model-select').change(changeModel);
    $("#total-examples").html(total_examples - 1);
    $('#switchToCustomMode').click(() => {
        window.location.href = 'custom_mode.html';
    });
    $('#page-input').keypress(function(event) {
        if (event.keyCode == 13) {
            gotobtn();
        }
    });
    $('#get-question-button').click(getAnswer);
    $('#download-table').click(downloadTable);
});