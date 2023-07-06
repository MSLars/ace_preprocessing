import itertools
import random
import re
from pathlib import Path

import spacy
import srsly
from bs4 import BeautifulSoup
from spacy.tokenizer import Tokenizer
from tqdm import tqdm

import ace_raw_data
import preprocessed_relation_classification_data
from data_split import extended
from preprocessing.data_model import RelationRepr, extract_entities, extract_times, extract_values, \
    extract_relations, tokenize_elem_and_mentions, ElementRepr

if __name__ == "__main__":
    INCLUDE_OTHER = False

    nlp = spacy.load("en_core_web_sm")

    special_cases = nlp.tokenizer.rules
    new_special_cases = {}
    for k, v in special_cases.items():
        match = re.match(r"([a-zA-Z]+)\.", k)
        if match and k.endswith(".") and len(v) == 1:

            new_v = {65: v[0][65][:-1]}
            if 67 in v[0]:
                new_v[67] = v[0][67][:-1]

            new_special_cases[k] = [new_v, {65: "."}]
        else:
            new_special_cases[k] = v

    new_special_cases["AIG.Greenberg"] = [{65: "AIG"}, {65: "."}, {65: "Greenberg"}]
    new_special_cases["its"] = [{65: "it"}, {65: "s"}]
    new_special_cases["worlds"] = [{65: "world"}, {65: "s"}]
    new_special_cases["exwife"] = [{65: "ex"}, {65: "wife"}]
    new_special_cases["exhusband"] = [{65: "ex"}, {65: "husband"}]
    new_special_cases["atman"] = [{65: "at"}, {65: "man"}]
    new_special_cases["drivers"] = [{65: "driver"}, {65: "s"}]
    new_special_cases["Indias"] = [{65: "India"}, {65: "s"}]
    new_special_cases["Tripuras"] = [{65: "Tripura"}, {65: "s"}]
    new_special_cases["ceos"] = [{65: "ceo"}, {65: "s"}]
    new_special_cases['U.S.-Kuwaiti'] = [{65: 'U.S.'}, {65: '-'}, {65: 'Kuwaiti'}]
    # ['Tripuras']'U.S.-made'
    suffixes = nlp.Defaults.suffixes + [r'''\.$''', r'''-$''']
    # suffixes = nlp.Defaults.suffixes + [r'''\.$''', r'''-$''']
    suffix_regex = spacy.util.compile_suffix_regex(suffixes)
    # nlp.tokenizer.suffix_search = suffix_regex.search

    # infixes = nlp.Defaults.infixes + [r'''(?<=[A-Za-z])\.(?=,)''', r'''\?''']
    infixes = nlp.Defaults.infixes + [r'''\?''']
    infix_regex = spacy.util.compile_infix_regex(infixes)
    # nlp.tokenizer.infix_finditer = infix_regex.finditer

    # prefixes = nlp.Defaults.prefixes + [r'''^~''', r'''^-''', r'''^[Uu]\.[Ss]''', r'''^\.''']
    prefixes = nlp.Defaults.prefixes + [r'''^~''', r'''^-''', r'''^[Uu]\.[Ss]''']
    prefix_regex = spacy.util.compile_prefix_regex(prefixes)

    simple_url_re = re.compile(r'''^https?://''')


    def custom_tokenizer(nlp):
        return Tokenizer(nlp.vocab,
                         rules=new_special_cases,
                         prefix_search=prefix_regex.search,
                         suffix_search=suffix_regex.search,
                         infix_finditer=infix_regex.finditer,
                         url_match=simple_url_re.match)


    # Add special case rule
    nlp.tokenizer = custom_tokenizer(nlp)

    nlp.disable_pipes("ner")

    raw_ace_path = Path(ace_raw_data.__path__[0])

    englich_data_path = raw_ace_path / "ace_2005_td_v7" / "data" / "English"

    final_samples = {
        "train": [],
        "dev": [],
        "test": []
    }

    # splits = srsly.read_json(Path(extended.__path__[0]) / "reduced_split.json")
    splits = srsly.read_json(Path(extended.__path__[0]) / "complete_split.json")

    all_sgm_files = [f for f in englich_data_path.rglob("*.sgm") if "timex2norm" in f.parts]

    for sgm_file in tqdm(all_sgm_files):

        t_n_ents = 0
        t_n_rels = 0

        t_n_ent_mentions = 0
        t_n_rel_mentions = 0

        t_n_sents = 0
        t_n_r_sents = 0

        t_n_seg_errors = 0
        t_n_tok_errors = 0

        t_n_toks = 0
        t_slens = []

        t_n_duplicate = 0

        annotation_file = sgm_file.with_suffix(".apf.xml")

        document_text = sgm_file.read_text()

        ## START Some ACE-BeautifulSoup specific preprocessing and parsing
        if "&amp;" in document_text:
            document_text = document_text.replace("&amp;", "&AmpA")

        if "<QUOTE PREVIOUSPOST=" in document_text:
            document_text = re.sub(r"<QUOTE PREVIOUSPOST=\"([^\"]*)\"/?>", "", document_text)

        if "</SUBJECT>\n\n<SUBJECT>" in document_text:
            document_text = document_text.replace("</SUBJECT>\n\n<SUBJECT>",
                                                  "</SUBJECT>--DOUBLE_N_PLACEHOLDER??<SUBJECT>")

        soup = BeautifulSoup(document_text, 'html.parser')

        complete_text = soup.getText()

        if "--DOUBLE_N_PLACEHOLDER??" in complete_text:
            complete_text = complete_text.replace("--DOUBLE_N_PLACEHOLDER??", "\n\n")

        ## END

        annotation_text = annotation_file.read_text()

        entities, id2entity, id2entity_mention = extract_entities(annotation_text, complete_text)

        mention_list = [(m.head_start, m.head_end, e.type) for e in entities for m in e.mentions]
        mention_set = set(mention_list)
        if len(mention_list) > len(mention_set):
            t_n_duplicate = len(mention_list) - len(mention_set)

        times, id2time, id2time_mention = extract_times(annotation_text)
        values, id2value, id2value_mention = extract_values(annotation_text)

        id2elem = id2time | id2entity | id2value
        id2elem_mention = id2time_mention | id2entity_mention | id2value_mention

        relations, id2relation, id2relation_mention = extract_relations(annotation_text, id2elem,
                                                                        id2elem_mention)

        doc = nlp(complete_text)

        for elem in relations:
            t_n_rels += 1
            for men in elem.mentions:
                t_n_rel_mentions += 1

        # for elem in entities + times + values:
        for elem in entities:
            err_cnt = tokenize_elem_and_mentions(elem, doc)
            t_n_tok_errors += err_cnt

        # This loop is only for debugging purposes
        # for elem in entities + times + values:
        for element in entities:
            t_n_ents += 1
            for mention in elem.mentions:
                t_n_ent_mentions += 1
                ent = doc[mention.head_token_start:mention.head_token_end + 1]

                char_text = complete_text[mention.head_start:mention.head_end + 1]
                token_text = ent.text

                if mention.head_text != char_text:
                    i = 1
                if mention.head_text != token_text:
                    i = 1
                if char_text != token_text:
                    i = 1

        t_n_toks += len(doc)

        t_slens = [len(s) for s in doc.sents]

        for sent in doc.sents:
            t_n_sents += 1
            sentence_key2mentions = {}
            sentence_relation_mentions = []

            sentence_char_entities = []
            sentence_token_entities = []

            sentence_element_mentions = []

            sentence_elements = set()

            # for element in entities + times + values:
            for element in entities:

                for mention in element.mentions:
                    if not (
                            mention.head_token_start >= sent.start and mention.head_token_end < sent.end):
                        continue

                    # Here we are shure, that the mention is in the current sentence
                    sentence_key2mentions[mention.id] = mention

                    entity_char_start = mention.head_start - sent.start_char
                    entity_char_end = mention.head_end - sent.start_char

                    entity_token_start = mention.head_token_start - sent.start
                    entity_token_end = mention.head_token_end - sent.start

                    entity_label = element.__class__.__name__
                    if hasattr(element, "type"):
                        entity_label = f"{element.type}"
                    if hasattr(element, "subtype"):
                        entity_label = f"{entity_label}_{element.subtype}"

                    ent_text = sent.text[entity_char_start:entity_char_end + 1]
                    ent_span = sent[entity_token_start:entity_token_end + 1]

                    sentence_char_entities.append(
                        [entity_char_start, entity_char_end, entity_label])
                    sentence_token_entities.append(
                        [entity_token_start, entity_token_end, entity_label])

                    sentence_element_mentions.append({
                        "id": mention.id,
                        "element": element.id,
                        "mention_type": mention.type if hasattr(mention, "type") else "VAL_REF",
                        "label": entity_label,
                    })

                    class_ = ""
                    if not hasattr(element, "class_"):
                        class_ = element.__class__.__name__
                    else:
                        class_ = element.class_

                    sentence_elements.add(ElementRepr(element.id, entity_label, class_))

            sentence_entities = [elem for elem in sentence_elements if "_" in elem.label]

            sentence_char_relations = []
            sentence_token_relations = []
            sentence_relation_mentions = []
            sentence_relations = set()

            if len(sentence_entities) >= 2:

                for arg1, arg2 in itertools.combinations(sentence_element_mentions, 2):
                    if not ("-E" in arg1["id"] and "-E" in arg2["id"]):
                        continue

                    all_relation_id_pairs = [{r.head.id, r.tail.id} for r in relations]

                    tmp_id_pair = {arg1["id"].split("-")[0], arg2["id"].split("-")[0]}

                    if tmp_id_pair in all_relation_id_pairs:
                        i = 10
                        continue

                    random_float = random.random()

                    if random_float < 0.01 and INCLUDE_OTHER:

                        tokens = [token.text for token in sent]

                        arg1_elem = sentence_key2mentions[arg1["id"]]
                        arg2_elem = sentence_key2mentions[arg2["id"]]

                        arg1_start_token = arg1_elem.head_token_start - sent.start
                        arg1_end_token = arg1_elem.head_token_end - sent.start

                        arg2_start_token = arg2_elem.head_token_start - sent.start
                        arg2_end_token = arg2_elem.head_token_end - sent.start

                        relation_tuple = (arg1_start_token,
                                          arg1_end_token,
                                          arg2_start_token,
                                          arg2_end_token,
                                          "OTHER")

                        for split, files in splits.items():
                            doc_key = sgm_file.stem

                            extend_start = min(arg1_start_token, arg2_start_token)
                            extend_end = max(arg1_end_token, arg2_end_token)

                            if doc_key in files:
                                final_samples[split].append({"tokens": tokens,
                                                             "relation": relation_tuple,
                                                             "extend": (extend_start, extend_end),
                                                             "lexicalcondition": "OTHER", })

            for relation in relations:
                for mention in relation.mentions:

                    if ((mention.arg1.id in sentence_key2mentions) and
                        (mention.arg2.id not in sentence_key2mentions)) or \
                            ((mention.arg2.id in sentence_key2mentions) and
                             (mention.arg1.id not in sentence_key2mentions)):
                        t_n_seg_errors += 1

                    if mention.arg1.id in sentence_key2mentions and mention.arg2.id in sentence_key2mentions:
                        t_n_r_sents += 1

                        relation_label = "UNDEFINED"
                        if hasattr(relation, "type"):
                            relation_label = relation.type
                        if hasattr(relation, "subtype"):
                            relation_label = f"{relation_label}_{relation.subtype}"

                        arg1_mention_head_text = mention.arg1.head_text
                        arg2_mention_head_text = mention.arg2.head_text

                        if "," in arg1_mention_head_text or "," in arg2_mention_head_text:
                            i = 10

                        arg1_start_char = mention.arg1.head_start - sent.start_char
                        arg1_end_char = mention.arg1.head_end - sent.start_char
                        arg1_start_token = mention.arg1.head_token_start - sent.start
                        arg1_end_token = mention.arg1.head_token_end - sent.start

                        arg1_tokens = [t.text for t in sent[arg1_start_token:arg1_end_token + 1]]

                        if arg1_mention_head_text.replace(" ", "").replace(".", "") != "".join(
                                [token for token in arg1_tokens if token != "AmpA"]).replace(".",
                                                                                             ""):
                            i = 10

                        arg2_start_char = mention.arg2.head_start - sent.start_char
                        arg2_end_char = mention.arg2.head_end - sent.start_char
                        arg2_start_token = mention.arg2.head_token_start - sent.start
                        arg2_end_token = mention.arg2.head_token_end - sent.start

                        arg2_tokens = [t.text for t in sent[arg2_start_token:arg2_end_token + 1]]

                        if arg2_mention_head_text.replace(" ", "").replace(".", "") != "".join(
                                [token for token in arg2_tokens
                                 if token != "AmpA"]).replace(" ", "").replace(".", ""):
                            i = 10

                        sentence_char_relations.append([arg1_start_char,
                                                        arg1_end_char,
                                                        arg2_end_char,
                                                        arg2_end_char,
                                                        relation_label])

                        sentence_token_relations.append([arg1_start_token,
                                                         arg1_end_token,
                                                         arg2_start_token,
                                                         arg2_end_token,
                                                         relation_label])

                        relation_head_text = sent.text[arg1_start_char:arg1_end_char + 1]
                        relation_head_span = sent[arg1_start_token:arg1_end_token + 1]

                        relation_tail_text = sent.text[arg2_start_char:arg2_end_char + 1]
                        relation_tail_span = sent[arg2_start_token:arg2_end_token + 1]

                        sentence_relation_mentions.append({
                            "id": mention.id,
                            "relation": relation.id,
                            "label": relation_label,
                            "lexicalcondition": mention.lexicalcondition,
                            "arg1": mention.arg1.id,
                            "arg2": mention.arg2.id,
                            "modality": relation.modality,
                            "tense": relation.tense,
                        })

                        sentence_relations.add(
                            RelationRepr(relation.id, relation_label, relation.modality,
                                         relation.tense))

                        tokens = [token.text for token in sent]

                        relation_tuple = (arg1_start_token,
                                          arg1_end_token,
                                          arg2_start_token,
                                          arg2_end_token,
                                          relation_label)

                        extend_span = doc.char_span(mention.extend_start, mention.extend_end + 1,
                                                    alignment_mode="expand")
                        extend_token_start = extend_span.start - sent.start
                        extend_token_end = extend_span.end - sent.start - 1

                        min_relation_token_start = min(arg1_start_token, arg2_start_token)
                        max_relation_token_end = max(arg1_end_token, arg2_end_token)

                        if extend_token_start > min_relation_token_start:
                            extend_token_start = min_relation_token_start
                        if extend_token_end < max_relation_token_end:
                            extend_token_end = max_relation_token_end

                        for split, files in splits.items():
                            doc_key = sgm_file.stem

                            if doc_key in files:

                                # Add some Information about argument1 and argument2
                                arg1_info = f"{mention.arg1.element.type}_{mention.arg1.element.subtype}_{mention.arg1.type}"
                                arg2_info = f"{mention.arg2.element.type}_{mention.arg2.element.subtype}_{mention.arg2.type}"

                                if mention.arg1.head_token_start > mention.arg2.head_token_start:
                                    arg1_info, arg2_info = arg2_info, arg1_info

                                final_samples[split].append({"tokens": tokens,
                                                             "relation": relation_tuple,
                                                             "extend": (
                                                                 extend_token_start,
                                                                 extend_token_end),
                                                             "lexicalcondition": mention.lexicalcondition,
                                                             "arg1_info": arg1_info,
                                                             "arg2_info": arg2_info,
                                                             "id": f"{doc_key}-{sent.start_char}"
                                                             })

        assigned = False
        for split, files in splits.items():
            doc_key = sgm_file.stem

            if doc_key in files:
                if assigned:
                    i = 10
                assigned = True

        if not assigned:
            i = 10

    for split, samples in final_samples.items():
        print(f"{split}: {len(samples)}")
        output_file = Path(
            preprocessed_relation_classification_data.__path__[0]) / f"{split}.jsonl"

        srsly.write_jsonl(output_file, samples)
