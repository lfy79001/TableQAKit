var split = 'dev';
var dataset = 'wikisql0';
var default_model = 'T5-small';
var model = 'T5-small';
var table_idx = 0;
var total_examples = 1;
var default_result = {};

function mod(n, m) {
    return ((n % m) + m) % m;
}

function changeSplit() {
    split = $('#split-select').val();
    table_idx = 0;

}

function changeDataset() {
    dataset = $('#dataset-select').val();
    table_idx = 0;
}

function changeDefaultModel() {
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

function changePictureHtml(pictures) {
    let pictures_html = ''
    for (let pic in pictures) {
        pictures_html += '<div class="col-sm-6 col-md-4 col-lg-3"><img src="../static/img/' + pic + '" style="max-width: 100%; height: auto;"></div>'
    }
    $('#image-container').html(pictures_html);
}

function changePropertiesHtml(properties) {
    $('#properties-container').html(properties);
}

function changeTextHtml(question) {
    $('#default-question').html(question);
}

function changeModelHtml(answer) {

}

function getData() {
    $.get('/table/default', { 'dataset_name': dataset, 'split': split, 'table_idx': table_idx },
        (data, status) => {
            log.console(status);
            log.console(data);
            total_examples = data.table_cnt
            default_result = data.generated_results
        })
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

function getAnswer(dataset, split, table_idx) {
    console.log('question:', $('#input-question').val());
    // TODO 自定义获取提问答案
}

$(document).ready(() => {
    $('#dataset-select').val(dataset);
    $('#splite-select').val(split);
    $('#default-model-select').val(default_model);
    $('#model-select').val(model);
    $("#dataset-select").change(changeDataset);
    $("#split-select").change(changeSplit);
    $('#default-model-select').change(changeDefaultModel);
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
});