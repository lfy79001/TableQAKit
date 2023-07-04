<template>
    <el-container style="width: 100%">
        <el-header style="width: 100%;">
            <el-row :gutter="20" style="width: 100%;">
                <el-col :span="8">
                    <div style="width: 100%;text-align: center; margin-top: 10px">TextTableQA</div>
                </el-col>
                <el-col :span="12">
                    <div>
                        <el-space wrap style="margin-top: 10px">
                            <el-button :icon="DArrowLeft" />
                            <el-button :icon="ArrowLeft" />
                            <el-input
                                v-model="jumpPage"
                                placeholder="Please Input"
                                :formatter="(value) => value.replace(/[^0-9]/ig, '')"
                                :parser="(value) => value.replace(/[^0-9]/ig, '')"
                                style="width: 10%"
                                clearable/>
                            <div>7699</div>
                            <el-button :icon="ArrowRight" style="width: 90%;"/>
                            <el-button :icon="DArrowRight" style="width: 90%;"/>
                        </el-space>
                    </div>
                </el-col>
                <el-col :span="4">
                    <div style="width: 100%;text-align: center; margin-top: 10px">TableQA Playground</div>
                </el-col>
            </el-row>
            <el-divider border-style="double" />
        </el-header>

        <el-container>
            <el-aside width="20%">
                <el-row class="left_row">
                    <div>Dataset</div>
                </el-row>
                <el-row class="left_row">
                    <el-select>
                        <el-option>cacapo</el-option>
                        <el-option>totto</el-option>
                        <el-option>e2e</el-option>
                    </el-select>
                </el-row>
                <el-row class="left_row">
                    <div>splite</div>
                </el-row>
                <el-row class="left_row">
                    <el-select>
                        <el-option>dev</el-option>
                        <el-option>test</el-option>
                        <el-option>train</el-option>
                    </el-select>
                </el-row>
                <el-row class="left_row">
                    <el-button type="primary">Download</el-button>
                </el-row>
                <el-row class="left_row">
                    <el-button type="primary">Upload Your Table</el-button>
                </el-row>
            </el-aside>
            <el-divider direction="vertical" style="height: 100%;"/>
            <el-main>
                <el-row :gutter="20">
                    <el-col :span="8">properties</el-col>
                    <el-col :span="16"><div /></el-col>
                </el-row>
                <el-row>
                    <el-button type="primary">reference</el-button>
                    <el-button type="primary">category</el-button>
                    <el-button type="primary">webnlg_id</el-button>
                </el-row>
                <el-card class="question-history">
                    <div v-for="o in 4" :key="o" class="text item">{{ 'List item ' + o }}</div>
                </el-card>
                <el-input
                    v-model="textarea"
                    :autosize="{minRows: 3, maxRows: 5}"
                    type="textarea"
                    placeholder="Please input your question"
                />
                <el-table :data="tableData" border style="width: 100%">
                    <el-table-column prop="date" label="Date" width="180" />
                    <el-table-column prop="name" label="Name" width="180" />
                    <el-table-column prop="address" label="Address" width="180"/>
                </el-table>
            </el-main>
            <el-divider direction="vertical" style="height: 100%;"/>
            <el-aside width="20%">
                <el-card class="box-card">
                    <template #header>
                        <div class="card-header">
                            <span>SQL</span>
                            <!--                            <el-button class="button" text>Operation button</el-button>-->
                        </div>
                    </template>
                    <div v-for="o in 4" :key="o" class="text item">{{ 'List item ' + o }}</div>
                </el-card>
                <el-card class="box-card">
                    <template #header>
                        <div class="card-header">
                            <span>Gold Answer</span>
                            <!--                            <el-button class="button" text>Operation button</el-button>-->
                        </div>
                    </template>
                    <div v-for="o in 4" :key="o" class="text item">{{ 'List item ' + o }}</div>
                </el-card>
                <el-card class="box-card">
                    <template #header>
                        <div class="card-header">
                            <span>T5-smallAnswer</span>
                            <!--                            <el-button class="button" text>Operation button</el-button>-->
                        </div>
                    </template>
                    <div v-for="o in 4" :key="o" class="text item">{{ 'List item ' + o }}</div>
                </el-card>
            </el-aside>
        </el-container>
        <el-footer>

        </el-footer>
    </el-container>
</template>

<script setup>
import {ArrowLeft, ArrowRight, DArrowLeft, DArrowRight} from '@element-plus/icons-vue'
const tableData = [
    {
        date: '2016-05-03',
        name: 'Tom',
        address: 'No. 189, Grove St, Los Angeles',
    },
    {
        date: '2016-05-02',
        name: 'Tom',
        address: 'No. 189, Grove St, Los Angeles',
    },
    {
        date: '2016-05-04',
        name: 'Tom',
        address: 'No. 189, Grove St, Los Angeles',
    },
    {
        date: '2016-05-01',
        name: 'Tom',
        address: 'No. 189, Grove St, Los Angeles',
    },
]

import { ref } from 'vue'
const jumpPage = ref('')
const textarea = ref('')
</script>

<script>

</script>

<style scoped>
.el-row {
    margin-bottom: 20px;
}
.el-card {
    margin-bottom: 10px;
}
</style>
