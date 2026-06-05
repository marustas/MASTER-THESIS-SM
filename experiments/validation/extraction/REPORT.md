# Extraction-pilot validation — report

## Per-document metrics

| doc_kind | doc_id | doc_title | gold_n | extracted_n | tp | fp | fn | near_miss | precision | recall | f1 | jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| programme | 21 | Game Development | 17 | 71 | 9 | 62 | 8 | 0 | 0.1268 | 0.5294 | 0.2045 | 0.1139 |
| programme | 25 | Information Systems Engineering | 16 | 56 | 2 | 54 | 14 | 0 | 0.0357 | 0.125 | 0.0556 | 0.0286 |
| programme | 29 | Informatics Engineering | 20 | 56 | 8 | 48 | 12 | 0 | 0.1429 | 0.4 | 0.2105 | 0.1176 |
| programme | 26 | Information Systems Engineering | 22 | 50 | 11 | 39 | 11 | 1 | 0.22 | 0.5 | 0.3056 | 0.1803 |
| programme | 12 | Informatics | 10 | 27 | 1 | 26 | 9 | 0 | 0.037 | 0.1 | 0.0541 | 0.0278 |
| job_ad | 253 | IT Vadovas (-ė) | 8 | 29 | 1 | 28 | 7 | 2 | 0.0345 | 0.125 | 0.0541 | 0.0278 |
| job_ad | 1 | Network Automation Engineer | 15 | 15 | 6 | 9 | 9 | 0 | 0.4 | 0.4 | 0.4 | 0.25 |
| job_ad | 429 | QA Engineer with verification | 9 | 9 | 2 | 7 | 7 | 0 | 0.2222 | 0.2222 | 0.2222 | 0.125 |
| job_ad | 193 | Ieškome Technologijų ir AI praktikanto (-ės). Gal tai tu? :) (Ukmergės g. 240 Vilnius) | 11 | 2 | 1 | 1 | 10 | 0 | 0.5 | 0.0909 | 0.1538 | 0.0833 |
| job_ad | 20 | BI programuotojas (-a) | 14 | 6 | 2 | 4 | 12 | 0 | 0.3333 | 0.1429 | 0.2 | 0.1111 |
| ALL | -1 | micro-average | 142 | 321 | 43 | 278 | 99 | 3 | 0.134 | 0.3028 | 0.1857 | 0.1065 |

## Error breakdown by document

### programme 21 (FP=62, TP=9, FN=8)

| doc_kind | doc_id | esco_uri | label | verdict | note |
| --- | --- | --- | --- | --- | --- |
| programme | 21 | http://data.europa.eu/esco/skill/54924a2c-daca-40d3-9716-4b38ceb04f38 | algorithms | TP |  |
| programme | 21 | http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97 | computer programming | TP |  |
| programme | 21 | http://data.europa.eu/esco/skill/afda6ca0-7aa8-41ef-ba5e-8dc607839daf | computer graphics | TP |  |
| programme | 21 | http://data.europa.eu/esco/skill/69bbd53f-fbb0-4476-b4b2-ef7844464e28 | web programming | TP |  |
| programme | 21 | http://data.europa.eu/esco/skill/4463a721-69f3-413d-8321-43e3af13a4f1 | use databases | TP |  |
| programme | 21 | http://data.europa.eu/esco/skill/97965983-0da4-4902-9daf-d5cd2693ef73 | 3D modelling | TP |  |
| programme | 21 | http://data.europa.eu/esco/skill/966f2fd3-3de6-42da-b87c-da924c6d7960 | digital game creation systems | TP |  |
| programme | 21 | http://data.europa.eu/esco/skill/d1a86399-24d8-415f-98b9-e8cbb6b04a26 | Unity (digital game creation systems) | TP |  |
| programme | 21 | http://data.europa.eu/esco/skill/a520b743-8f40-43ac-a2d5-755899120844 | audio technology | TP |  |
| programme | 21 | http://data.europa.eu/esco/skill/e6ee5bb8-12b9-4e60-8baf-102d0c4f1da5 | semantics | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/658605f2-1c95-49f0-bd98-0af7b15ad0b0 | entrepreneurship | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/3cd569a2-4f88-4c1e-9995-8dce8c5e51a7 | JavaScript | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/df64a7cc-2e96-4304-95d3-56fe0ac2dd39 | design thinking | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/57231a22-4da7-49c8-97b8-75672feadf1e | manage quantitative data | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/519e801b-3cc4-44d4-bcf1-32fdb9a77e51 | personal development | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/4707da90-9cfc-46ca-8de0-38a0b7bfb137 | think analytically | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/143769cb-b61e-47d8-a61e-eedfbec1016c | business intelligence | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/0823ccef-813f-4f22-afef-ac0d68615e8f | computer simulation | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/2180bd8c-86de-4889-8165-adac902eee9d | embedded systems | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/bfe4f330-d595-48c7-ab3c-f309471d6953 | psychology | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/ff8bd17c-77e9-4d4e-929a-a0cf86f3ed34 | surveillance methods | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/2636b3d3-843e-46a9-8b4c-a9d6ca3f5a2d | provide technical documentation | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/d56e2c2f-3e01-4b76-a7bb-ecb6d430172f | think holistically | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/e747e77e-0ea1-4001-8b07-1d11946b5f1b | French | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/094ef6aa-844f-44ce-8456-fdc49276bf58 | digital printing | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/76ef6ed3-1658-4a1a-9593-204d799c6d0c | NoSQL | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/b105ec9b-0857-41d6-8d07-a83e58b73d90 | ICT system programming | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/198b614d-a9af-46f5-b3fd-2672b401dd8e | health and safety in the workplace | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/ed5bbc64-7017-4e2e-b44e-65df88013a84 | analyse information systems | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/6891bbce-20bf-4afc-bd5e-75bdf54c0165 | Russian | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/f8e3425c-fe44-4ffb-bafe-0e20d91dadf4 | SAP Data Services | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/4812a4ea-dc55-4dc6-b9b0-4a59bba2c647 | German | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/7ee51257-8947-4b69-9bdf-322baf0a6398 | Swedish | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/6dc2dfac-3e21-44dd-a71c-9a1c8fe2514c | make independent operating decisions | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/48db96bf-3314-45c6-bad8-fdb6e20e5639 | data engineering | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/7afb5a64-e574-421a-bb3a-7a7bc108d2a5 | perform warehousing operations | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/14ee9f76-3524-43d5-8a1a-5ba8283f8bd7 | Spanish | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/123cf7c5-6a3b-419d-be4f-49a6cc020f9f | interactive media | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/9e84a506-df06-4be3-874a-fa01293e3dd5 | business processes | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/4c58528e-bdaa-43ad-8f5a-8ad0b8cd4bbb | design principles | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/93350886-a21b-4e61-a1b1-319a325c0f90 | express yourself creatively | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/9136cbf1-7916-4f1c-bc9a-0318ee1d6016 | human-computer interaction | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/a88da605-095f-4a4f-be1c-d1444df8228a | digital media | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/7369f779-4b71-4aab-8836-48b69c676eec | operate relational database management system | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/791e2eea-3380-4aed-b996-4bcabbe88591 | market participants | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/1cca610d-2afc-44a7-97fc-f2262fb5fc75 | surveying | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/a4d336a6-9ffd-402a-91cc-f359716ba4e0 | ML (computer programming) | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/9ef0f3a0-9ce2-4ef1-a987-0366b5cb2dbe | database development tools | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/d62d2b4c-a6f8-439e-8a1b-4f29ab5f2c47 | make decisions | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/b17f4305-741a-4a1c-8fe8-6f11cb3d5c0a | business communication | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/ab1e97ed-2319-4293-a8b7-072d2648822f | database management systems | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/a59708e3-e654-4e37-8b8a-741c3b756eee | multimedia systems | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/b633eb55-8f1f-4ae6-ab4c-2022ffe2cb7f | C++ | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/31b67516-af16-4b97-8430-a8a8e0f84190 | assessment processes | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/15d76317-c71a-4fa2-aadc-2ecc34e627b7 | communication | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/f4b3b063-02df-4a92-bb0e-d01e45642c6f | in-circuit test | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/4e097377-7950-4993-9e22-a61b067b5c00 | social sciences | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/3ec2e4d6-7000-4905-bf1a-c5b1679416de | data warehouse | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/848b5879-10af-4e3a-bfbf-263956b4ebf3 | management department processes | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/9effa3d7-c0c4-4583-ad94-b496ba5e5f2c | data mining methods | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/1bba98a7-92b9-450b-9235-e0c905f8f3c4 | information architecture | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/db33c0f3-43ee-4ba3-ba47-9269ac837697 | critically evaluate information and its sources | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/49fed129-e32b-4f67-b80a-609d79e45b20 | electronic and telecommunication equipment | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/35f1dcdc-577f-46c6-96d9-6c7a64501de9 | laboratory equipment | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/609a8ac1-9d29-4237-9886-596dbbe7ca8a | address an audience | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/6fe019dd-027f-45f3-b19c-d27f7ae00980 | assess students | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/ddc3119d-1d6e-4324-9125-a3380d299ac5 | computer technology | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/1fe42b38-cd42-4fc7-ae77-14e4c9e96295 | Chinese | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/1258cc12-37bb-4a12-b219-9c3d6b294533 | work in an organised manner | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/91dd8397-063d-499c-83b9-0603a10d94ac | writing techniques | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/7b5cce4d-c7fe-4119-b48f-70aa05391787 | computer science | FP |  |
| programme | 21 | http://data.europa.eu/esco/skill/d9013e0e-e937-43d5-ab71-0e917ee882b8 | manage time | FN |  |
| programme | 21 | http://data.europa.eu/esco/skill/75d8e5d9-bef3-418b-9011-01bff9f27207 | lead others | FN |  |
| programme | 21 | http://data.europa.eu/esco/skill/6a322874-e32f-4cd8-9683-badce67a7f73 | develop automated software tests | FN |  |
| programme | 21 | http://data.europa.eu/esco/skill/60c78287-22eb-4103-9c8c-28deaa460da0 | work in teams | FN |  |
| programme | 21 | http://data.europa.eu/esco/skill/cfa2be0d-96d5-4017-a866-962efb9c5070 | signal processing | FN |  |
| programme | 21 | http://data.europa.eu/esco/skill/43ae58b9-5e56-4524-b45a-b422777a0576 | database | FN |  |
| programme | 21 | http://data.europa.eu/esco/skill/29fb0fb5-dfc4-4098-ac9b-3a712000f48f | manage database | FN |  |
| programme | 21 | http://data.europa.eu/esco/skill/52e53619-fa77-4f72-b237-5e4aae784dc2 | financial management | FN |  |

### programme 25 (FP=54, FN=14, TP=2)

| doc_kind | doc_id | esco_uri | label | verdict | note |
| --- | --- | --- | --- | --- | --- |
| programme | 25 | http://data.europa.eu/esco/skill/95c35c3a-035f-47c2-90cf-7e934d20fc08 | electronics | TP |  |
| programme | 25 | http://data.europa.eu/esco/skill/b633eb55-8f1f-4ae6-ab4c-2022ffe2cb7f | C++ | TP |  |
| programme | 25 | http://data.europa.eu/esco/skill/d04ee340-5378-4601-8181-19da6d5cbfe0 | manage website | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/e6ee5bb8-12b9-4e60-8baf-102d0c4f1da5 | semantics | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/377861af-f966-4c87-a2db-7e99904312b9 | geographic information systems | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/8fdf4273-d8ce-47a2-b461-45cb1282ef36 | reverse engineering | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/88af9294-b697-4687-ae19-aefbb6234dfa | economics | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/dba46f87-0831-49cd-a1c7-340a653c0221 | Agile development | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/63b9a087-a5db-424e-96ef-3212c8b5311e | tutor students | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/4707da90-9cfc-46ca-8de0-38a0b7bfb137 | think analytically | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/692ace70-7f97-4214-92bd-88f7637d8a44 | physics | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/49de9958-2aa4-4eef-a89d-fe5d5bcd28c4 | adapt to change | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/0823ccef-813f-4f22-afef-ac0d68615e8f | computer simulation | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/bfe4f330-d595-48c7-ab3c-f309471d6953 | psychology | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/209a5498-3449-4689-8ed9-bd08cab4fd78 | engineering principles | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/76ef6ed3-1658-4a1a-9593-204d799c6d0c | NoSQL | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/f8e3425c-fe44-4ffb-bafe-0e20d91dadf4 | SAP Data Services | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/74e49482-75e8-4637-bdb1-fff868d02f7a | scientific literature | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/fbca4e43-3816-44cf-81a5-d7b5ae320c00 | guide staff | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/9983816d-cc78-4d3f-9e3c-c7baa9ebc77a | computer equipment | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/001115fb-569f-4ee6-8381-c6807ef2527f | show initiative | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/3d64b8fd-bb09-4d13-a3cf-300ed8909088 | write specifications | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/045f71e6-0699-4169-8a54-9c6b96f3174d | advise others | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/6eff134b-e34f-4d6e-a6e8-5e47cf2228d0 | risk management | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/c624c6a3-b0ba-4a31-a296-0d433fe47e41 | think creatively | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/d9013e0e-e937-43d5-ab71-0e917ee882b8 | manage time | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/b3950b87-a980-4cd4-a795-be8a9b63661d | Lithuanian | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/d9a3cb06-d6e0-4a64-88a6-9a8e11a99c93 | labour market | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/9a58cd26-58eb-4a1c-b1b6-64037fe9cfa1 | think abstractly | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/93350886-a21b-4e61-a1b1-319a325c0f90 | express yourself creatively | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/7cf2762d-32e1-4b21-8ef8-574e40310c18 | morality | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/e49f4158-9d4c-425d-bf32-dfe89b19840a | plan | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/7ee4c2ea-b349-4bd2-81a3-ec31475d4833 | statistics | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/08e2ff26-88ff-47b5-9d37-d19ace01b075 | manage feedback | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/adc6dc11-3376-467b-96c5-9b0a21edc869 | solve problems | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/a4d336a6-9ffd-402a-91cc-f359716ba4e0 | ML (computer programming) | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/afda6ca0-7aa8-41ef-ba5e-8dc607839daf | computer graphics | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/6d3edede-8951-4621-a835-e04323300fa0 | English | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/f9a6f35b-01a7-40c9-8b61-b6ee46f97272 | operating systems | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/15d76317-c71a-4fa2-aadc-2ecc34e627b7 | communication | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/31b67516-af16-4b97-8430-a8a8e0f84190 | assessment processes | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/967b60c2-4657-4ffc-bcaf-aab565793f97 | philosophy | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/a7f0fbe0-c546-4f30-8e41-34a58c64567e | data storage | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/a571ae14-3e16-4fd3-a615-5646e0b0b696 | inspect data | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/0ab9d433-10e5-4683-ae54-4687179a5259 | literature | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/fecf8a0d-62c4-4e71-9b03-0f4fc2ad7bf5 | data models | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/8881a9c2-bd07-4954-bf27-c1f8acca9af0 | sociology | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/9af1bd12-01bd-4d6a-947b-69e64f23150a | process qualitative information | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/89f6560b-2194-45c9-9ece-d33049a73eef | computer engineering | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/49fed129-e32b-4f67-b80a-609d79e45b20 | electronic and telecommunication equipment | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/35f1dcdc-577f-46c6-96d9-6c7a64501de9 | laboratory equipment | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/4339176e-3acd-4f7f-a5d9-445bee3d23f2 | mathematics | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/ddc3119d-1d6e-4324-9125-a3380d299ac5 | computer technology | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/732cb4dd-af91-4c88-9a9f-5fc62144e500 | provide information | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/48db96bf-3314-45c6-bad8-fdb6e20e5639 | data engineering | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/7b5cce4d-c7fe-4119-b48f-70aa05391787 | computer science | FP |  |
| programme | 25 | http://data.europa.eu/esco/skill/2164e860-7f20-48bc-b98c-5d9f8a561550 | design computer network | FN |  |
| programme | 25 | http://data.europa.eu/esco/skill/8088750d-8388-4170-a76f-48354c469c44 | cyber security | FN |  |
| programme | 25 | http://data.europa.eu/esco/skill/02058de6-4b98-449f-8a45-8588b0eb2446 | network engineering | FN |  |
| programme | 25 | http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97 | computer programming | FN |  |
| programme | 25 | http://data.europa.eu/esco/skill/a8d07b5a-c1a1-42c6-9d53-db9c7a2ca996 | PostgreSQL | FN |  |
| programme | 25 | http://data.europa.eu/esco/skill/69bbd53f-fbb0-4476-b4b2-ef7844464e28 | web programming | FN |  |
| programme | 25 | http://data.europa.eu/esco/skill/4463a721-69f3-413d-8321-43e3af13a4f1 | use databases | FN |  |
| programme | 25 | http://data.europa.eu/esco/skill/3f86173d-e101-4fcd-934f-ff9de29c081c | perform system analysis | FN |  |
| programme | 25 | http://data.europa.eu/esco/skill/b8c01891-e3df-4a4b-948d-95b45e1788f5 | mobile device software frameworks | FN |  |
| programme | 25 | http://data.europa.eu/esco/skill/598de5b0-5b58-4ea7-8058-a4bc4d18c742 | SQL | FN |  |
| programme | 25 | http://data.europa.eu/esco/skill/fae27053-8924-4bfd-b565-c9fe502044c9 | system design | FN |  |
| programme | 25 | http://data.europa.eu/esco/skill/43ae58b9-5e56-4524-b45a-b422777a0576 | database | FN |  |
| programme | 25 | http://data.europa.eu/esco/skill/29fb0fb5-dfc4-4098-ac9b-3a712000f48f | manage database | FN |  |
| programme | 25 | http://data.europa.eu/esco/skill/94dd823c-148e-4614-a6e8-99249b16357d | ICT project management | FN |  |

### programme 29 (FP=48, FN=12, TP=8)

| doc_kind | doc_id | esco_uri | label | verdict | note |
| --- | --- | --- | --- | --- | --- |
| programme | 29 | http://data.europa.eu/esco/skill/8088750d-8388-4170-a76f-48354c469c44 | cyber security | TP |  |
| programme | 29 | http://data.europa.eu/esco/skill/adc6dc11-3376-467b-96c5-9b0a21edc869 | solve problems | TP |  |
| programme | 29 | http://data.europa.eu/esco/skill/88af9294-b697-4687-ae19-aefbb6234dfa | economics | TP |  |
| programme | 29 | http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97 | computer programming | TP |  |
| programme | 29 | http://data.europa.eu/esco/skill/f049d050-12da-4e40-813a-2b5eb6df6b51 | Internet of Things | TP |  |
| programme | 29 | http://data.europa.eu/esco/skill/ddc3119d-1d6e-4324-9125-a3380d299ac5 | computer technology | TP |  |
| programme | 29 | http://data.europa.eu/esco/skill/54924a2c-daca-40d3-9716-4b38ceb04f38 | algorithms | TP |  |
| programme | 29 | http://data.europa.eu/esco/skill/967b60c2-4657-4ffc-bcaf-aab565793f97 | philosophy | TP |  |
| programme | 29 | http://data.europa.eu/esco/skill/e6ee5bb8-12b9-4e60-8baf-102d0c4f1da5 | semantics | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/3cd569a2-4f88-4c1e-9995-8dce8c5e51a7 | JavaScript | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/86d2e2ea-1ba2-4aa6-b465-8a1f9abc81b8 | apply information security policies | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/57231a22-4da7-49c8-97b8-75672feadf1e | manage quantitative data | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/c29aa9d2-4da8-4bdd-831c-8d4a2fb51730 | work independently | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/dba46f87-0831-49cd-a1c7-340a653c0221 | Agile development | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/d8829a1d-dbde-435b-b921-29d6462f35c9 | Android (mobile operating systems) | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/abdc7ac8-151f-40c6-bc1a-1e9b4b073290 | augmented reality | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/692ace70-7f97-4214-92bd-88f7637d8a44 | physics | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/143769cb-b61e-47d8-a61e-eedfbec1016c | business intelligence | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/0823ccef-813f-4f22-afef-ac0d68615e8f | computer simulation | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/ff8bd17c-77e9-4d4e-929a-a0cf86f3ed34 | surveillance methods | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/adcca622-b933-4ba3-bc43-bf371879edc3 | data security principles | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/2180bd8c-86de-4889-8165-adac902eee9d | embedded systems | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/198b614d-a9af-46f5-b3fd-2672b401dd8e | health and safety in the workplace | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/ed5bbc64-7017-4e2e-b44e-65df88013a84 | analyse information systems | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/f8e3425c-fe44-4ffb-bafe-0e20d91dadf4 | SAP Data Services | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/209a5498-3449-4689-8ed9-bd08cab4fd78 | engineering principles | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/76ef6ed3-1658-4a1a-9593-204d799c6d0c | NoSQL | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/6fa1c2c0-a012-4ca0-9642-e01569ba322c | ICT system integration | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/001115fb-569f-4ee6-8381-c6807ef2527f | show initiative | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/9983816d-cc78-4d3f-9e3c-c7baa9ebc77a | computer equipment | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/045f71e6-0699-4169-8a54-9c6b96f3174d | advise others | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/ecc4552a-92c5-4222-b18d-faf5ac841080 | deep learning | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/c624c6a3-b0ba-4a31-a296-0d433fe47e41 | think creatively | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/b3950b87-a980-4cd4-a795-be8a9b63661d | Lithuanian | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/fbdbed1c-442a-47b3-a26f-0e867d82bbea | ergonomics | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/9136cbf1-7916-4f1c-bc9a-0318ee1d6016 | human-computer interaction | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/7ee4c2ea-b349-4bd2-81a3-ec31475d4833 | statistics | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/e49f4158-9d4c-425d-bf32-dfe89b19840a | plan | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/294cf13e-4fdf-4cb8-bb6b-31b9da7f4819 | logic | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/a4d336a6-9ffd-402a-91cc-f359716ba4e0 | ML (computer programming) | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/afda6ca0-7aa8-41ef-ba5e-8dc607839daf | computer graphics | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/9ef0f3a0-9ce2-4ef1-a987-0366b5cb2dbe | database development tools | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/edebd83d-35f6-4ed5-a940-6c203d178c01 | data science | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/6d3edede-8951-4621-a835-e04323300fa0 | English | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/b633eb55-8f1f-4ae6-ab4c-2022ffe2cb7f | C++ | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/f9a6f35b-01a7-40c9-8b61-b6ee46f97272 | operating systems | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/15d76317-c71a-4fa2-aadc-2ecc34e627b7 | communication | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/1bba98a7-92b9-450b-9235-e0c905f8f3c4 | information architecture | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/a571ae14-3e16-4fd3-a615-5646e0b0b696 | inspect data | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/db33c0f3-43ee-4ba3-ba47-9269ac837697 | critically evaluate information and its sources | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/49fed129-e32b-4f67-b80a-609d79e45b20 | electronic and telecommunication equipment | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/35f1dcdc-577f-46c6-96d9-6c7a64501de9 | laboratory equipment | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/4339176e-3acd-4f7f-a5d9-445bee3d23f2 | mathematics | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/48db96bf-3314-45c6-bad8-fdb6e20e5639 | data engineering | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/7b5cce4d-c7fe-4119-b48f-70aa05391787 | computer science | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/b758675a-b03b-4e4b-897c-57ca14b1a3d0 | probability theory | FP |  |
| programme | 29 | http://data.europa.eu/esco/skill/2164e860-7f20-48bc-b98c-5d9f8a561550 | design computer network | FN |  |
| programme | 29 | http://data.europa.eu/esco/skill/02058de6-4b98-449f-8a45-8588b0eb2446 | network engineering | FN |  |
| programme | 29 | http://data.europa.eu/esco/skill/4463a721-69f3-413d-8321-43e3af13a4f1 | use databases | FN |  |
| programme | 29 | http://data.europa.eu/esco/skill/2450c3b3-e78e-435b-b84d-e05d984e71dc | software architecture models | FN |  |
| programme | 29 | http://data.europa.eu/esco/skill/bd14968e-e409-45af-b362-3495ed7b10e0 | cloud technologies | FN |  |
| programme | 29 | http://data.europa.eu/esco/skill/f0de4973-0a70-4644-8fd4-3a97080476f4 | DevOps | FN |  |
| programme | 29 | http://data.europa.eu/esco/skill/fae27053-8924-4bfd-b565-c9fe502044c9 | system design | FN |  |
| programme | 29 | http://data.europa.eu/esco/skill/43ae58b9-5e56-4524-b45a-b422777a0576 | database | FN |  |
| programme | 29 | http://data.europa.eu/esco/skill/29fb0fb5-dfc4-4098-ac9b-3a712000f48f | manage database | FN |  |
| programme | 29 | http://data.europa.eu/esco/skill/1d86f05e-e9cc-40ce-99d8-2b21cc71b16b | solution deployment | FN |  |
| programme | 29 | http://data.europa.eu/esco/skill/bf6c5ed4-84af-440f-abcc-7fa5ba19c738 | real-time computing | FN |  |
| programme | 29 | http://data.europa.eu/esco/skill/f7e2eb04-3e50-4561-bce1-7e51a1fec308 | define software architecture | FN |  |

### programme 26 (FP=38, TP=11, FN=11, NEAR_MISS=1)

| doc_kind | doc_id | esco_uri | label | verdict | note |
| --- | --- | --- | --- | --- | --- |
| programme | 26 | http://data.europa.eu/esco/skill/8088750d-8388-4170-a76f-48354c469c44 | cyber security | TP |  |
| programme | 26 | http://data.europa.eu/esco/skill/43ae58b9-5e56-4524-b45a-b422777a0576 | database | TP |  |
| programme | 26 | http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97 | computer programming | TP |  |
| programme | 26 | http://data.europa.eu/esco/skill/913e7e83-b8f8-4574-b1ca-1b38f3fd974a | execute software tests | TP |  |
| programme | 26 | http://data.europa.eu/esco/skill/89f6560b-2194-45c9-9ece-d33049a73eef | computer engineering | TP |  |
| programme | 26 | http://data.europa.eu/esco/skill/bd14968e-e409-45af-b362-3495ed7b10e0 | cloud technologies | TP |  |
| programme | 26 | http://data.europa.eu/esco/skill/60c78287-22eb-4103-9c8c-28deaa460da0 | work in teams | TP |  |
| programme | 26 | http://data.europa.eu/esco/skill/f049d050-12da-4e40-813a-2b5eb6df6b51 | Internet of Things | TP |  |
| programme | 26 | http://data.europa.eu/esco/skill/f9a6f35b-01a7-40c9-8b61-b6ee46f97272 | operating systems | TP |  |
| programme | 26 | http://data.europa.eu/esco/skill/15d76317-c71a-4fa2-aadc-2ecc34e627b7 | communication | TP |  |
| programme | 26 | http://data.europa.eu/esco/skill/54924a2c-daca-40d3-9716-4b38ceb04f38 | algorithms | TP |  |
| programme | 26 | http://data.europa.eu/esco/skill/1349751e-dfb5-492e-89af-e7d9bd2546ac | patents | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/88af9294-b697-4687-ae19-aefbb6234dfa | economics | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/c29aa9d2-4da8-4bdd-831c-8d4a2fb51730 | work independently | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/dba46f87-0831-49cd-a1c7-340a653c0221 | Agile development | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/172020d1-e151-445b-8173-e2a5fb16fe51 | utilise computer-aided software engineering tools | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/d8829a1d-dbde-435b-b921-29d6462f35c9 | Android (mobile operating systems) | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/63b9a087-a5db-424e-96ef-3212c8b5311e | tutor students | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/abdc7ac8-151f-40c6-bc1a-1e9b4b073290 | augmented reality | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/692ace70-7f97-4214-92bd-88f7637d8a44 | physics | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/49de9958-2aa4-4eef-a89d-fe5d5bcd28c4 | adapt to change | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/2180bd8c-86de-4889-8165-adac902eee9d | embedded systems | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/adcca622-b933-4ba3-bc43-bf371879edc3 | data security principles | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/bfe4f330-d595-48c7-ab3c-f309471d6953 | psychology | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/209a5498-3449-4689-8ed9-bd08cab4fd78 | engineering principles | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/6fa1c2c0-a012-4ca0-9642-e01569ba322c | ICT system integration | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/001115fb-569f-4ee6-8381-c6807ef2527f | show initiative | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/9983816d-cc78-4d3f-9e3c-c7baa9ebc77a | computer equipment | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/045f71e6-0699-4169-8a54-9c6b96f3174d | advise others | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/ecc4552a-92c5-4222-b18d-faf5ac841080 | deep learning | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/c624c6a3-b0ba-4a31-a296-0d433fe47e41 | think creatively | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/3e40c7d0-0e36-4b33-bc33-0aa87eda0561 | electrical engineering | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/fbdbed1c-442a-47b3-a26f-0e867d82bbea | ergonomics | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/9136cbf1-7916-4f1c-bc9a-0318ee1d6016 | human-computer interaction | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/7ee4c2ea-b349-4bd2-81a3-ec31475d4833 | statistics | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/a4346013-a967-4a58-a533-6b32ad1364c5 | data protection | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/08e2ff26-88ff-47b5-9d37-d19ace01b075 | manage feedback | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/adc6dc11-3376-467b-96c5-9b0a21edc869 | solve problems | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/75d8e5d9-bef3-418b-9011-01bff9f27207 | lead others | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/edebd83d-35f6-4ed5-a940-6c203d178c01 | data science | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/b17f4305-741a-4a1c-8fe8-6f11cb3d5c0a | business communication | NEAR_MISS | matches gold http://data.europa.eu/esco/skill/15d76317-c71a-4fa2-aadc-2ecc34e627b7 (label Jaccard 0.50) |
| programme | 26 | http://data.europa.eu/esco/skill/31b67516-af16-4b97-8430-a8a8e0f84190 | assessment processes | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/95c35c3a-035f-47c2-90cf-7e934d20fc08 | electronics | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/a571ae14-3e16-4fd3-a615-5646e0b0b696 | inspect data | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/0ab9d433-10e5-4683-ae54-4687179a5259 | literature | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/8881a9c2-bd07-4954-bf27-c1f8acca9af0 | sociology | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/397da142-ab35-48fe-b154-7c38f447adfb | digital systems | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/9af1bd12-01bd-4d6a-947b-69e64f23150a | process qualitative information | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/4339176e-3acd-4f7f-a5d9-445bee3d23f2 | mathematics | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/e87ec79a-c9ff-46f5-84fa-7a0f394cdf40 | robotics | FP |  |
| programme | 26 | http://data.europa.eu/esco/skill/b07daddc-8625-4360-946e-ad2b0e56ebf6 | read engineering drawings | FN |  |
| programme | 26 | http://data.europa.eu/esco/skill/2164e860-7f20-48bc-b98c-5d9f8a561550 | design computer network | FN |  |
| programme | 26 | http://data.europa.eu/esco/skill/02058de6-4b98-449f-8a45-8588b0eb2446 | network engineering | FN |  |
| programme | 26 | http://data.europa.eu/esco/skill/4463a721-69f3-413d-8321-43e3af13a4f1 | use databases | FN |  |
| programme | 26 | http://data.europa.eu/esco/skill/6a322874-e32f-4cd8-9683-badce67a7f73 | develop automated software tests | FN |  |
| programme | 26 | http://data.europa.eu/esco/skill/29fb0fb5-dfc4-4098-ac9b-3a712000f48f | manage database | FN |  |
| programme | 26 | http://data.europa.eu/esco/skill/5d74614d-32c8-4ca4-9818-6980e52424b1 | plan software testing | FN |  |
| programme | 26 | http://data.europa.eu/esco/skill/24c200e5-be20-4370-a137-ab53797f3a17 | software frameworks | FN |  |
| programme | 26 | http://data.europa.eu/esco/skill/ddc3119d-1d6e-4324-9125-a3380d299ac5 | computer technology | FN |  |
| programme | 26 | http://data.europa.eu/esco/skill/59ea80e1-463a-4dba-82c6-d0b6d577d532 | technical drawings | FN |  |
| programme | 26 | http://data.europa.eu/esco/skill/7111b95d-0ce3-441a-9d92-4c75d05c4388 | project management | FN |  |

### programme 12 (FP=26, FN=9, TP=1)

| doc_kind | doc_id | esco_uri | label | verdict | note |
| --- | --- | --- | --- | --- | --- |
| programme | 12 | http://data.europa.eu/esco/skill/7b5cce4d-c7fe-4119-b48f-70aa05391787 | computer science | TP |  |
| programme | 12 | http://data.europa.eu/esco/skill/d04ee340-5378-4601-8181-19da6d5cbfe0 | manage website | FP |  |
| programme | 12 | http://data.europa.eu/esco/skill/dba46f87-0831-49cd-a1c7-340a653c0221 | Agile development | FP |  |
| programme | 12 | http://data.europa.eu/esco/skill/63b9a087-a5db-424e-96ef-3212c8b5311e | tutor students | FP |  |
| programme | 12 | http://data.europa.eu/esco/skill/4707da90-9cfc-46ca-8de0-38a0b7bfb137 | think analytically | FP |  |
| programme | 12 | http://data.europa.eu/esco/skill/49de9958-2aa4-4eef-a89d-fe5d5bcd28c4 | adapt to change | FP |  |
| programme | 12 | http://data.europa.eu/esco/skill/5bbaa0e6-0fd7-4df2-9db7-34f78b40dc34 | marketing management | FP |  |
| programme | 12 | http://data.europa.eu/esco/skill/209a5498-3449-4689-8ed9-bd08cab4fd78 | engineering principles | FP |  |
| programme | 12 | http://data.europa.eu/esco/skill/fbca4e43-3816-44cf-81a5-d7b5ae320c00 | guide staff | FP |  |
| programme | 12 | http://data.europa.eu/esco/skill/9983816d-cc78-4d3f-9e3c-c7baa9ebc77a | computer equipment | FP |  |
| programme | 12 | http://data.europa.eu/esco/skill/3d64b8fd-bb09-4d13-a3cf-300ed8909088 | write specifications | FP |  |
| programme | 12 | http://data.europa.eu/esco/skill/11c56452-fcec-4b00-9695-cca4728e5048 | marketing analytics | FP |  |
| programme | 12 | http://data.europa.eu/esco/skill/045f71e6-0699-4169-8a54-9c6b96f3174d | advise others | FP |  |
| programme | 12 | http://data.europa.eu/esco/skill/b3950b87-a980-4cd4-a795-be8a9b63661d | Lithuanian | FP |  |
| programme | 12 | http://data.europa.eu/esco/skill/9a58cd26-58eb-4a1c-b1b6-64037fe9cfa1 | think abstractly | FP |  |
| programme | 12 | http://data.europa.eu/esco/skill/e49f4158-9d4c-425d-bf32-dfe89b19840a | plan | FP |  |
| programme | 12 | http://data.europa.eu/esco/skill/08e2ff26-88ff-47b5-9d37-d19ace01b075 | manage feedback | FP |  |
| programme | 12 | http://data.europa.eu/esco/skill/adc6dc11-3376-467b-96c5-9b0a21edc869 | solve problems | FP |  |
| programme | 12 | http://data.europa.eu/esco/skill/d62d2b4c-a6f8-439e-8a1b-4f29ab5f2c47 | make decisions | FP |  |
| programme | 12 | http://data.europa.eu/esco/skill/b633eb55-8f1f-4ae6-ab4c-2022ffe2cb7f | C++ | FP |  |
| programme | 12 | http://data.europa.eu/esco/skill/31b67516-af16-4b97-8430-a8a8e0f84190 | assessment processes | FP |  |
| programme | 12 | http://data.europa.eu/esco/skill/a571ae14-3e16-4fd3-a615-5646e0b0b696 | inspect data | FP |  |
| programme | 12 | http://data.europa.eu/esco/skill/fecf8a0d-62c4-4e71-9b03-0f4fc2ad7bf5 | data models | FP |  |
| programme | 12 | http://data.europa.eu/esco/skill/1bba98a7-92b9-450b-9235-e0c905f8f3c4 | information architecture | FP |  |
| programme | 12 | http://data.europa.eu/esco/skill/9af1bd12-01bd-4d6a-947b-69e64f23150a | process qualitative information | FP |  |
| programme | 12 | http://data.europa.eu/esco/skill/ddc3119d-1d6e-4324-9125-a3380d299ac5 | computer technology | FP |  |
| programme | 12 | http://data.europa.eu/esco/skill/a6532ae5-efc4-4ca4-b172-c11471940d09 | electronic business | FP |  |
| programme | 12 | http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97 | computer programming | FN |  |
| programme | 12 | http://data.europa.eu/esco/skill/4463a721-69f3-413d-8321-43e3af13a4f1 | use databases | FN |  |
| programme | 12 | http://data.europa.eu/esco/skill/913e7e83-b8f8-4574-b1ca-1b38f3fd974a | execute software tests | FN |  |
| programme | 12 | http://data.europa.eu/esco/skill/5d74614d-32c8-4ca4-9818-6980e52424b1 | plan software testing | FN |  |
| programme | 12 | http://data.europa.eu/esco/skill/24c200e5-be20-4370-a137-ab53797f3a17 | software frameworks | FN |  |
| programme | 12 | http://data.europa.eu/esco/skill/f9a6f35b-01a7-40c9-8b61-b6ee46f97272 | operating systems | FN |  |
| programme | 12 | http://data.europa.eu/esco/skill/43ae58b9-5e56-4524-b45a-b422777a0576 | database | FN |  |
| programme | 12 | http://data.europa.eu/esco/skill/29fb0fb5-dfc4-4098-ac9b-3a712000f48f | manage database | FN |  |
| programme | 12 | http://data.europa.eu/esco/skill/54924a2c-daca-40d3-9716-4b38ceb04f38 | algorithms | FN |  |

### job_ad 253 (FP=26, FN=7, NEAR_MISS=2, TP=1)

| doc_kind | doc_id | esco_uri | label | verdict | note |
| --- | --- | --- | --- | --- | --- |
| job_ad | 253 | http://data.europa.eu/esco/skill/8088750d-8388-4170-a76f-48354c469c44 | cyber security | TP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/86d2e2ea-1ba2-4aa6-b465-8a1f9abc81b8 | apply information security policies | FP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/c29aa9d2-4da8-4bdd-831c-8d4a2fb51730 | work independently | FP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/dc9a236c-c640-43c3-812f-269403591edb | customer relationship management | FP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/b0288eea-74b3-460a-8ac5-edc4ba70b75c | manage closed-circuit television system | FP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/ecc18804-a466-40d9-98b4-fba5cd67dd4b | accounting | FP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/7954861c-86d4-4529-afbb-2c23dab9ac74 | negotiate compromises | FP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/f8e3425c-fe44-4ffb-bafe-0e20d91dadf4 | SAP Data Services | FP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/6fa1c2c0-a012-4ca0-9642-e01569ba322c | ICT system integration | FP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/6891bbce-20bf-4afc-bd5e-75bdf54c0165 | Russian | FP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/66da3162-6c8d-44cb-9dd5-6f9efe1fcf67 | trademarks | FP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/3c03ee71-4a23-448f-b79e-81fd75d27dca | apply change management | FP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/6eff134b-e34f-4d6e-a6e8-5e47cf2228d0 | risk management | NEAR_MISS | matches gold http://data.europa.eu/esco/skill/69cfc5ed-6569-4aca-a4cc-fd782ba51d9c (label Jaccard 0.50) |
| job_ad | 253 | http://data.europa.eu/esco/skill/f0de4973-0a70-4644-8fd4-3a97080476f4 | DevOps | FP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/7afb5a64-e574-421a-bb3a-7a7bc108d2a5 | perform warehousing operations | FP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/9e84a506-df06-4be3-874a-fa01293e3dd5 | business processes | FP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/0a48d064-dd04-47fb-a00d-85e9b4874033 | business management principles | FP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/b3950b87-a980-4cd4-a795-be8a9b63661d | Lithuanian | FP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/bec4359e-cb92-468f-a997-8fb28e32fba9 | ICT project management methodologies | NEAR_MISS | matches gold http://data.europa.eu/esco/skill/7111b95d-0ce3-441a-9d92-4c75d05c4388 (label Jaccard 0.50) |
| job_ad | 253 | http://data.europa.eu/esco/skill/e49f4158-9d4c-425d-bf32-dfe89b19840a | plan | FP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/ddbf25f1-b91a-4083-9ef1-cc113a46e4c0 | ecosystems | FP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/d47b9f28-8131-4efd-8801-53f226955f21 | periodisation | FP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/598de5b0-5b58-4ea7-8058-a4bc4d18c742 | SQL | FP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/bdcf429c-5ccf-4c3d-bb61-4c987573a35e | show entrepreneurial spirit | FP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/633a3637-2c6b-40ae-ac38-289eb2a62aa6 | business analysis | FP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/60c78287-22eb-4103-9c8c-28deaa460da0 | work in teams | FP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/ddc3119d-1d6e-4324-9125-a3380d299ac5 | computer technology | FP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/f84a433f-34f1-4083-b0a3-24802623509c | web services | FP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/7b5cce4d-c7fe-4119-b48f-70aa05391787 | computer science | FP |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/69cfc5ed-6569-4aca-a4cc-fd782ba51d9c | implement ICT risk management | FN |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/7111b95d-0ce3-441a-9d92-4c75d05c4388 | project management | FN |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97 | computer programming | FN |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/24c200e5-be20-4370-a137-ab53797f3a17 | software frameworks | FN |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/3c76296d-4bbd-44ba-8eaa-95bf275f79b7 | manage ICT project | FN |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/afdb41dd-5dd5-4fbb-bf9e-e041422698b4 | coordinate operational activities | FN |  |
| job_ad | 253 | http://data.europa.eu/esco/skill/3579208e-49b3-4ce4-98e7-20e41b1ce8d4 | develop information security strategy | FN |  |

### job_ad 1 (FP=9, FN=9, TP=6)

| doc_kind | doc_id | esco_uri | label | verdict | note |
| --- | --- | --- | --- | --- | --- |
| job_ad | 1 | http://data.europa.eu/esco/skill/02058de6-4b98-449f-8a45-8588b0eb2446 | network engineering | TP |  |
| job_ad | 1 | http://data.europa.eu/esco/skill/334e3e49-fb02-4051-809a-f06adfdc1c40 | troubleshoot | TP |  |
| job_ad | 1 | http://data.europa.eu/esco/skill/3e40c7d0-0e36-4b33-bc33-0aa87eda0561 | electrical engineering | TP |  |
| job_ad | 1 | http://data.europa.eu/esco/skill/9d2e926f-53d9-41f5-98f3-19dfaa687f3f | tools for software configuration management | TP |  |
| job_ad | 1 | http://data.europa.eu/esco/skill/ccd0a1d9-afda-43d9-b901-96344886e14d | Python (computer programming) | TP |  |
| job_ad | 1 | http://data.europa.eu/esco/skill/7b5cce4d-c7fe-4119-b48f-70aa05391787 | computer science | TP |  |
| job_ad | 1 | http://data.europa.eu/esco/skill/055192dc-f16c-4855-835d-19cac6ff20aa | life sciences | FP |  |
| job_ad | 1 | http://data.europa.eu/esco/skill/b8df1689-6763-437d-bfca-fe03d43f36c3 | Computer Assisted Language Learning | FP |  |
| job_ad | 1 | http://data.europa.eu/esco/skill/e5ad0cce-e3d5-4c60-8504-5d1c7e3b55b9 | electrical equipment components | FP |  |
| job_ad | 1 | http://data.europa.eu/esco/skill/75b30aeb-34c0-40f4-b77d-271d75a98b14 | improve business processes | FP |  |
| job_ad | 1 | http://data.europa.eu/esco/skill/584c13e8-fd06-4ca3-a45b-cca5b2f147a7 | Portuguese | FP |  |
| job_ad | 1 | http://data.europa.eu/esco/skill/6d3edede-8951-4621-a835-e04323300fa0 | English | FP |  |
| job_ad | 1 | http://data.europa.eu/esco/skill/14ee9f76-3524-43d5-8a1a-5ba8283f8bd7 | Spanish | FP |  |
| job_ad | 1 | http://data.europa.eu/esco/skill/c624c6a3-b0ba-4a31-a296-0d433fe47e41 | think creatively | FP |  |
| job_ad | 1 | http://data.europa.eu/esco/skill/bec4359e-cb92-468f-a997-8fb28e32fba9 | ICT project management methodologies | FP |  |
| job_ad | 1 | http://data.europa.eu/esco/skill/2164e860-7f20-48bc-b98c-5d9f8a561550 | design computer network | FN |  |
| job_ad | 1 | http://data.europa.eu/esco/skill/adc6dc11-3376-467b-96c5-9b0a21edc869 | solve problems | FN |  |
| job_ad | 1 | http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97 | computer programming | FN |  |
| job_ad | 1 | http://data.europa.eu/esco/skill/55514865-3066-4abd-86a9-dbe45a440882 | root cause analysis | FN |  |
| job_ad | 1 | http://data.europa.eu/esco/skill/2450c3b3-e78e-435b-b84d-e05d984e71dc | software architecture models | FN |  |
| job_ad | 1 | http://data.europa.eu/esco/skill/bd14968e-e409-45af-b362-3495ed7b10e0 | cloud technologies | FN |  |
| job_ad | 1 | http://data.europa.eu/esco/skill/f0de4973-0a70-4644-8fd4-3a97080476f4 | DevOps | FN |  |
| job_ad | 1 | http://data.europa.eu/esco/skill/bf6c5ed4-84af-440f-abcc-7fa5ba19c738 | real-time computing | FN |  |
| job_ad | 1 | http://data.europa.eu/esco/skill/f7e2eb04-3e50-4561-bce1-7e51a1fec308 | define software architecture | FN |  |

### job_ad 429 (FP=7, FN=7, TP=2)

| doc_kind | doc_id | esco_uri | label | verdict | note |
| --- | --- | --- | --- | --- | --- |
| job_ad | 429 | http://data.europa.eu/esco/skill/6d3edede-8951-4621-a835-e04323300fa0 | English | TP |  |
| job_ad | 429 | http://data.europa.eu/esco/skill/598de5b0-5b58-4ea7-8058-a4bc4d18c742 | SQL | TP |  |
| job_ad | 429 | http://data.europa.eu/esco/skill/580660a6-5d3a-421d-a54f-d85b706c2b2f | use online tools to collaborate | FP |  |
| job_ad | 429 | http://data.europa.eu/esco/skill/0e1fe34b-f4e7-4642-8c8b-5a05ac3438e5 | manage schedule of tasks | FP |  |
| job_ad | 429 | http://data.europa.eu/esco/skill/10c42c72-ecaa-414c-a014-7e97fcecae8d | reprography | FP |  |
| job_ad | 429 | http://data.europa.eu/esco/skill/15d76317-c71a-4fa2-aadc-2ecc34e627b7 | communication | FP |  |
| job_ad | 429 | http://data.europa.eu/esco/skill/ddc3119d-1d6e-4324-9125-a3380d299ac5 | computer technology | FP |  |
| job_ad | 429 | http://data.europa.eu/esco/skill/c624c6a3-b0ba-4a31-a296-0d433fe47e41 | think creatively | FP |  |
| job_ad | 429 | http://data.europa.eu/esco/skill/7a8fb784-67fa-41e9-a75c-6b491d91f800 | develop strategy to solve problems | FP |  |
| job_ad | 429 | http://data.europa.eu/esco/skill/85f46538-ae70-498a-bfbc-b8ddafe96c7d | levels of software testing | FN |  |
| job_ad | 429 | http://data.europa.eu/esco/skill/913e7e83-b8f8-4574-b1ca-1b38f3fd974a | execute software tests | FN |  |
| job_ad | 429 | http://data.europa.eu/esco/skill/5d74614d-32c8-4ca4-9818-6980e52424b1 | plan software testing | FN |  |
| job_ad | 429 | http://data.europa.eu/esco/skill/cffc3e97-e942-4b13-a2f3-0bf4910c06d3 | use technical documentation | FN |  |
| job_ad | 429 | http://data.europa.eu/esco/skill/327d15c8-29ff-4ad4-a4fe-6536d777a45f | documentation types | FN |  |
| job_ad | 429 | http://data.europa.eu/esco/skill/2636b3d3-843e-46a9-8b4c-a9d6ca3f5a2d | provide technical documentation | FN |  |
| job_ad | 429 | http://data.europa.eu/esco/skill/e207163b-7963-4c3e-9494-7a4bb000211b | estimate duration of work | FN |  |

### job_ad 193 (FN=10, TP=1, FP=1)

| doc_kind | doc_id | esco_uri | label | verdict | note |
| --- | --- | --- | --- | --- | --- |
| job_ad | 193 | http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97 | computer programming | TP |  |
| job_ad | 193 | http://data.europa.eu/esco/skill/ddc3119d-1d6e-4324-9125-a3380d299ac5 | computer technology | FP |  |
| job_ad | 193 | http://data.europa.eu/esco/skill/f4a6e9f7-5cff-46c0-894c-59c20bb78694 | automation technology | FN |  |
| job_ad | 193 | http://data.europa.eu/esco/skill/913e7e83-b8f8-4574-b1ca-1b38f3fd974a | execute software tests | FN |  |
| job_ad | 193 | http://data.europa.eu/esco/skill/bd14968e-e409-45af-b362-3495ed7b10e0 | cloud technologies | FN |  |
| job_ad | 193 | http://data.europa.eu/esco/skill/24c200e5-be20-4370-a137-ab53797f3a17 | software frameworks | FN |  |
| job_ad | 193 | http://data.europa.eu/esco/skill/5d74614d-32c8-4ca4-9818-6980e52424b1 | plan software testing | FN |  |
| job_ad | 193 | http://data.europa.eu/esco/skill/4707da90-9cfc-46ca-8de0-38a0b7bfb137 | think analytically | FN |  |
| job_ad | 193 | http://data.europa.eu/esco/skill/cffc3e97-e942-4b13-a2f3-0bf4910c06d3 | use technical documentation | FN |  |
| job_ad | 193 | http://data.europa.eu/esco/skill/e465a154-93f7-4973-9ce1-31659fe16dd2 | principles of artificial intelligence | FN |  |
| job_ad | 193 | http://data.europa.eu/esco/skill/327d15c8-29ff-4ad4-a4fe-6536d777a45f | documentation types | FN |  |
| job_ad | 193 | http://data.europa.eu/esco/skill/2636b3d3-843e-46a9-8b4c-a9d6ca3f5a2d | provide technical documentation | FN |  |

### job_ad 20 (FN=12, FP=4, TP=2)

| doc_kind | doc_id | esco_uri | label | verdict | note |
| --- | --- | --- | --- | --- | --- |
| job_ad | 20 | http://data.europa.eu/esco/skill/598de5b0-5b58-4ea7-8058-a4bc4d18c742 | SQL | TP |  |
| job_ad | 20 | http://data.europa.eu/esco/skill/143769cb-b61e-47d8-a61e-eedfbec1016c | business intelligence | TP |  |
| job_ad | 20 | http://data.europa.eu/esco/skill/ddc3119d-1d6e-4324-9125-a3380d299ac5 | computer technology | FP |  |
| job_ad | 20 | http://data.europa.eu/esco/skill/7ee4c2ea-b349-4bd2-81a3-ec31475d4833 | statistics | FP |  |
| job_ad | 20 | http://data.europa.eu/esco/skill/4339176e-3acd-4f7f-a5d9-445bee3d23f2 | mathematics | FP |  |
| job_ad | 20 | http://data.europa.eu/esco/skill/15d76317-c71a-4fa2-aadc-2ecc34e627b7 | communication | FP |  |
| job_ad | 20 | http://data.europa.eu/esco/skill/c3e36d05-8ae8-447f-bb2b-6f9409f85389 | deliver visual presentation of data | FN |  |
| job_ad | 20 | http://data.europa.eu/esco/skill/adc6dc11-3376-467b-96c5-9b0a21edc869 | solve problems | FN |  |
| job_ad | 20 | http://data.europa.eu/esco/skill/21d2f96d-35f7-4e3f-9745-c533d2dd6e97 | computer programming | FN |  |
| job_ad | 20 | http://data.europa.eu/esco/skill/97bd1c21-66b2-4b7e-ad0f-e3cda590e378 | data analytics | FN |  |
| job_ad | 20 | http://data.europa.eu/esco/skill/2b92a5b2-6758-4ee3-9fb4-b6387a55cc8f | perform data analysis | FN |  |
| job_ad | 20 | http://data.europa.eu/esco/skill/98a1dec3-8138-4f46-a596-5e2a83b884b9 | business analytics | FN |  |
| job_ad | 20 | http://data.europa.eu/esco/skill/633a3637-2c6b-40ae-ac38-289eb2a62aa6 | business analysis | FN |  |
| job_ad | 20 | http://data.europa.eu/esco/skill/24c200e5-be20-4370-a137-ab53797f3a17 | software frameworks | FN |  |
| job_ad | 20 | http://data.europa.eu/esco/skill/cffc3e97-e942-4b13-a2f3-0bf4910c06d3 | use technical documentation | FN |  |
| job_ad | 20 | http://data.europa.eu/esco/skill/65e58886-bd1e-4c5b-8ca5-8d9b353c8aa1 | data visualisation software | FN |  |
| job_ad | 20 | http://data.europa.eu/esco/skill/327d15c8-29ff-4ad4-a4fe-6536d777a45f | documentation types | FN |  |
| job_ad | 20 | http://data.europa.eu/esco/skill/2636b3d3-843e-46a9-8b4c-a9d6ca3f5a2d | provide technical documentation | FN |  |
