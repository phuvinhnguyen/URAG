import json as js
import os

if __name__ == "__main__":
    # dataset_path = ['/media/volume/LLMRag/URAG/datasets/commit_message_qa.json', '/media/volume/LLMRag/URAG/datasets/crag_task_1_and_2_mcqa.json',
    # '/media/volume/LLMRag/URAG/datasets/dialfact.json', '/media/volume/LLMRag/URAG/datasets/healthver_mcqa.json',
    # '/media/volume/LLMRag/URAG/datasets/multinewsum_mcqa.json', '/media/volume/LLMRag/URAG/datasets/odex.json',
    # '/media/volume/LLMRag/URAG/datasets/OlympiadBench.json', '/media/volume/LLMRag/URAG/datasets/scifact_mcqa.json']
    # result = {}
    # for file in dataset_path:
    #     dataset = js.load(open(os.path.join(os.path.dirname(__file__), file), "r"))
    #     count_num_ans = {}
    #     count_question_word = {}
    #     for question in dataset['calibration']:
    #         if len(question['options']) not in count_num_ans:
    #             count_num_ans[len(question['options'])] = 0
    #         count_num_ans[len(question['options'])] += 1
    #         if question['question'].split(' ')[0] not in count_question_word:
    #             count_question_word[question['question'].split(' ')[0]] = 0
    #         count_question_word[question['question'].split(' ')[0]] += 1
    #     count_num_ans_test = {}
    #     count_question_word_test = {}
    #     for question in dataset['test']:
    #         if len(question['options']) not in count_num_ans_test:
    #             count_num_ans_test[len(question['options'])] = 0
    #         count_num_ans_test[len(question['options'])] += 1
    #         if question['question'].split(' ')[0] not in count_question_word_test:
    #             count_question_word_test[question['question'].split(' ')[0]] = 0
    #         count_question_word_test[question['question'].split(' ')[0]] += 1
    #     result[file] = {
    #         'description': dataset['description'],
    #         'total_samples': dataset['total_samples'],
    #         'calibration_samples': dataset['calibration_samples'],
    #         'test_samples': dataset['test_samples'],
    #         'count_num_ans_test': {key: value for key, value in sorted(count_num_ans_test.items(), key=lambda item: item[1], reverse=True)},
    #         'count_question_word_test': {key: value for key, value in sorted(count_question_word_test.items(), key=lambda item: item[1], reverse=True)},
    #         'count_num_ans_calibration': {key: value for key, value in sorted(count_num_ans.items(), key=lambda item: item[1], reverse=True)},
    #         'count_question_word_calibration': {key: value for key, value in sorted(count_question_word.items(), key=lambda item: item[1], reverse=True)}
    #     }
    # js.dump(result, open(os.path.join(os.path.dirname(__file__), "result.json"), "w"), indent=4)

    #EDA CRAG only
    dataset_path = '/media/volume/LLMRag/URAG/datasets/crag_task_1_and_2_mcqa.json'
    result = {}
    dataset = js.load(open(os.path.join(os.path.dirname(__file__), dataset_path), "r"))
    domain = {}
    question_type = {}
    for ques in dataset['calibration']:
        if ques['domain'] not in domain:
            domain[ques['domain']] = 0
        domain[ques['domain']] += 1
        if ques['question_type'] not in question_type:
            question_type[ques['question_type']] = 0
        question_type[ques['question_type']] += 1
    for ques in dataset['test']:
        if ques['domain'] not in domain:
            domain[ques['domain']] = 0
        domain[ques['domain']] += 1
        if ques['question_type'] not in question_type:
            question_type[ques['question_type']] = 0
        question_type[ques['question_type']] += 1
    result['domain'] = domain
    result['question_type'] = question_type
    js.dump(result, open(os.path.join(os.path.dirname(__file__), "result.json"), "w"), indent=4)