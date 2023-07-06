from pydantic import BaseModel
from spacy.tokens import Span

from bs4 import BeautifulSoup
from overrides import overrides
from dataclasses import dataclass
from typing import Union, Optional

from spacy.tokens import Doc

OFFSET_CORECTION_KEYS = {
    'FLOPPINGACES_20041114.1240.039': 1,
    'MARKETVIEW_20050206.2009': 4, }


@dataclass
class ElementRepr:
    id: str
    label: str
    class_: str

    def __hash__(self):
        return hash(self.id + self.label + self.class_)

    def __eq__(self, other):
        return self.id == other.id and self.label == other.label and self.class_ == other.class_

    def __repr__(self):
        return f'{self.id} {self.label} {self.class_}'

    def to_json(self):
        return {
            'id': self.id,
            'label': self.label,
            'class': self.class_,
        }


@dataclass
class Element:
    id: str
    min_extend_char_idx: int
    max_extend_char_idx: int
    mentions: list
    repr_text: str = None
    min_extend_token_idx: Optional[int] = None
    max_extend_token_idx: Optional[int] = None

    def to_json(self):
        return {
            "id": self.id,
            "min_extend_char_idx": self.min_extend_char_idx,
            "max_extend_char_idx": self.max_extend_char_idx,
            "mentions": [mention.id for mention in self.mentions],
            "repr_text": self.repr_text,
            "min_extend_token_idx": self.min_extend_token_idx,
            "max_extend_token_idx": self.max_extend_token_idx,
        }

    # TODO: This may not be good enough when applied to other than original data
    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


@dataclass
class ElementMention:
    id: str
    element: Union[Element, int]
    extend_start: int
    extend_end: int
    head_text: str = None
    head_start: Optional[int] = None
    head_end: Optional[int] = None
    extend_token_start: Optional[int] = None
    extend_token_end: Optional[int] = None
    head_token_start: Optional[int] = None
    head_token_end: Optional[int] = None

    def to_json(self):
        return {
            "id": self.id,
            "element": self.element.id if isinstance(self.element, Element) else self.element,
            "extend_start": self.extend_start,
            "extend_end": self.extend_end,
            "head_start": self.head_start,
            "head_end": self.head_end,
            "extend_token_start": self.extend_token_start,
            "extend_token_end": self.extend_token_end,
            "head_token_start": self.head_token_start,
            "head_token_end": self.head_token_end,
        }

    # TODO: This may not be good enough when applied to other than original data
    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


def get_char_span(start_char_idx: int,
                  end_char_idx: int,
                  doc: Doc,
                  ent_label: str = None,
                  shift: bool = False):
    """

    :param shift:
    :param ent_label:
    :param end_char_idx:
    :param start_char_idx:
    :param doc: Spacy Doc. Used to handle the transformation from char to token indices
    :return:
    """
    error = False

    if ent_label is None:
        ent_label = "Extend"

    mention_head_span = doc.char_span(start_char_idx, end_char_idx + 1)

    if mention_head_span is None and shift:
        mention_head_span = doc.char_span(start_char_idx - 1, end_char_idx)
        error = True

    if mention_head_span is None:
        mention_head_span = doc.char_span(start_char_idx, end_char_idx + 1, alignment_mode="expand")
        error = True

    return mention_head_span, error


def extraxt_label(mention):
    res = "Value"
    if hasattr(mention.element, "type"):
        res = f"{mention.element.type}"
    if hasattr(mention.element, "subtype"):
        res += f"_{mention.element.subtype}"
    elif mention.element.__class__.__name__ == "TimeX2":
        res = "TIMEX2"
    return res


def tokenize_elem_and_mentions(elem: Element, doc: Doc, shift: bool = False):
    elem_extend, _ = get_char_span(elem.min_extend_char_idx, elem.max_extend_char_idx, doc)

    elem.min_extend_token_idx = elem_extend.start
    elem.max_extend_token_idx = elem_extend.end - 1

    err_cnt = 0

    for mention in elem.mentions:

        ent_label = extraxt_label(mention)

        mention_extend, _ = get_char_span(mention.extend_start, mention.extend_end, doc, ent_label)
        mention_head, error = get_char_span(mention.head_start, mention.head_end, doc, ent_label,
                                            shift)

        if error:
            err_cnt += 1

        mention.extend_token_start = mention_extend.start
        mention.extend_token_end = mention_extend.end - 1

        mention.head_token_start = mention_head.start
        mention.head_token_end = mention_head.end - 1

    return err_cnt


class Sentence(BaseModel):
    id: str

    start: int
    end: int

    token_start: int
    token_end: int

    text: str
    tokens: list[str]

    entity_mentions: list = None
    relation_mentions: list = None

    timex2_mentions: list = None
    valueMentions: list = None


@dataclass
class RelationRepr:
    id: str
    label: str
    modality: str
    tense: str

    def __hash__(self):
        return hash(self.id + self.label + self.modality + self.tense)

    def __eq__(self, other):
        return self.id == other.id and self.label == other.label and \
            self.modality == other.modality and self.tense == other.tense

    def __repr__(self):
        return f"RelationRepr(id={self.id}, label={self.label}, modality={self.modality}, " \
               f"tense={self.tense})"

    def to_json(self):
        return {
            'id': self.id,
            'label': self.label,
            'modality': self.modality,
            'tense': self.tense
        }


@dataclass
class Relation(Element):
    type: str = None
    subtype: str = None
    modality: str = None
    tense: str = None
    head: Element = None
    tail: Element = None

    @overrides
    def to_json(self):
        return {
            'id': self.id,
            'type': self.type,
            'subtype': self.subtype,
            'modality': self.modality,
            'tense': self.tense,
            'head': self.head.id,
            'tail': self.tail.id,
            'min_extend_char_idx': self.min_extend_char_idx,
            'max_extend_char_idx': self.max_extend_char_idx,
            'mentions': [mention.id for mention in self.mentions]
        }

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


@dataclass
class RelationMention(ElementMention):
    lexicalcondition: str = None
    arg1: ElementMention = None
    arg2: ElementMention = None

    @overrides
    def to_json(self):
        return {
            'id': self.id,
            'element': self.element.id,
            'lexicalcondition': self.lexicalcondition,
            'arg1': self.arg1.id,
            'arg2': self.arg2.id,
            'extend_start': self.extend_start,
            'extend_end': self.extend_end,
        }

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


def extract_relations(annotation_text, id2entity, id2entity_mention):
    soup = BeautifulSoup(annotation_text, 'lxml')
    relations = []
    id2relation_mention = {}

    for relation in soup.find_all("relation"):

        relation_id = relation.attrs['id']
        relation_type = relation.attrs['type']
        relation_subtype = relation.attrs['subtype'] if 'subtype' in relation.attrs else ""
        relation_modality = relation.attrs['modality'] if 'modality' in relation.attrs else ""
        relation_tense = relation.attrs['tense'] if 'tense' in relation.attrs else ""

        head = None
        tail = None

        for relation_argument in relation.find_all("relation_argument"):

            argument_entity = id2entity[relation_argument.attrs['refid']]

            if relation_argument.attrs['role'] == 'Arg-1':
                head = argument_entity
            elif relation_argument.attrs['role'] == 'Arg-2':
                tail = argument_entity

        relation_min_extend_char_idx = 10000000
        relation_max_extend_char_idx = 0

        relation_mentions = []

        for relation_mention in relation.find_all("relation_mention"):
            relation_mention_id = relation_mention.attrs['id']
            relation_mention_lexicalcondition = relation_mention.attrs[
                'lexicalcondition'] if 'lexicalcondition' in relation_mention.attrs else ""

            extend = relation_mention.find("extent", recursive=False)

            extend_start = int(extend.find("charseq").attrs['start'])
            extend_end = int(extend.find("charseq").attrs['end'])

            head_mention = None
            tail_mention = None

            for relation_mention_argument in relation_mention.find_all("relation_mention_argument"):
                argument_entity_mention = id2entity_mention[
                    relation_mention_argument.attrs['refid']]

                if relation_mention_argument.attrs['role'] == 'Arg-1':
                    head_mention = argument_entity_mention
                elif relation_mention_argument.attrs['role'] == 'Arg-2':
                    tail_mention = argument_entity_mention

            relation_min_extend_char_idx = min(relation_min_extend_char_idx, extend_start)
            relation_max_extend_char_idx = max(relation_max_extend_char_idx, extend_end)

            relation_mentions.append(RelationMention(
                id=relation_mention_id,
                element=relation_id,
                lexicalcondition=relation_mention_lexicalcondition,
                extend_start=extend_start,
                extend_end=extend_end,
                arg1=head_mention,
                arg2=tail_mention,
            ))

        relation = Relation(
            id=relation_id,
            type=relation_type,
            subtype=relation_subtype,
            modality=relation_modality,
            tense=relation_tense,
            min_extend_char_idx=relation_min_extend_char_idx,
            max_extend_char_idx=relation_max_extend_char_idx,
            mentions=relation_mentions,
            head=head,
            tail=tail
        )

        for relation_mention in relation_mentions:
            relation_mention.element = relation
            id2relation_mention[relation_mention.id] = relation_mention

        relations.append(relation)

    id2relation = {relation.id: relation for relation in relations}

    return relations, id2relation, id2relation_mention


@dataclass
class Value(Element):
    type: str = None
    sub_type: str = None

    @overrides
    def to_json(self):
        return {
            'id': self.id,
            'type': self.type,
            'sub_type': self.sub_type,
            'min_extend_char_idx': self.min_extend_char_idx,
            'max_extend_char_idx': self.max_extend_char_idx,
            'mentions': [mention.id for mention in self.mentions]
        }

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


@dataclass
class ValueMention(ElementMention):

    @overrides
    def to_json(self):
        return {
            'id': self.id,
            'element': self.element.id,
            'extend_start': self.extend_start,
            'extend_end': self.extend_end,
            'head_start': self.head_start,
            'head_end': self.head_end,
            'head_text': self.head_text
        }

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


def extract_values(annotation_text):
    soup = BeautifulSoup(annotation_text, 'lxml')
    values = []
    id2value_mention = {}

    for value in soup.find_all("value"):

        value_id = value.attrs['id']
        value_type = value.attrs['type'] if 'type' in value.attrs else None
        value_sub_type = value.attrs['sub_type'] if 'sub_type' in value.attrs else None

        value_mentions = []

        for value_mention in value.find_all("value_mention"):
            value_mention_id = value_mention.attrs['id']

            extend = value_mention.find("extent", recursive=False)

            extend_start = int(extend.find("charseq", recursive=False).attrs['start'])
            extend_end = int(extend.find("charseq", recursive=False).attrs['end'])

            value_text = value_mention.find("charseq").text

            value_mentions.append(ValueMention(
                id=value_mention_id,
                element=value_id,
                extend_start=extend_start,
                extend_end=extend_end,
                head_start=extend_start,
                head_end=extend_end,
                head_text=value_text
            ))

        value = Value(
            id=value_id,
            type=value_type,
            sub_type=value_sub_type,
            min_extend_char_idx=min([mention.extend_start for mention in value_mentions]),
            max_extend_char_idx=max([mention.extend_end for mention in value_mentions]),
            mentions=value_mentions
        )

        for value_mention in value_mentions:
            value_mention.element = value
            id2value_mention[value_mention.id] = value_mention

        values.append(value)

    id2value = {value.id: value for value in values}

    return values, id2value, id2value_mention


@dataclass
class TimeX2(Element):
    val: str = None
    anchor_val: str = None
    anchor_dir: str = None

    @overrides
    def to_json(self):
        return {
            'id': self.id,
            'val': self.val,
            'anchor_val': self.anchor_val,
            'anchor_dir': self.anchor_dir,
            'min_extend_char_idx': self.min_extend_char_idx,
            'max_extend_char_idx': self.max_extend_char_idx,
            'mentions': [mention.id for mention in self.mentions]
        }

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


@dataclass
class TimeX2Mention(ElementMention):

    @overrides
    def to_json(self):
        return {
            'id': self.id,
            'element': self.element.id,
            'extend_start': self.extend_start,
            'extend_end': self.extend_end,
            'head_start': self.head_start,
            'head_end': self.head_end,
            'head_text': self.head_text
        }

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


def extract_times(annotation_text):
    soup = BeautifulSoup(annotation_text, 'lxml')
    times = []
    id2time_mention = {}

    for timex2 in soup.find_all("timex2"):

        timex2_id = timex2.attrs['id']
        timex2_val = timex2.attrs['val'] if 'val' in timex2.attrs else None
        timex2_anchor_val = timex2.attrs['anchor_val'] if 'anchor_val' in timex2.attrs else None
        timex2_anchor_dir = timex2.attrs['anchor_dir'] if 'anchor_dir' in timex2.attrs else None

        time_mentions = []

        for timex2_mention in timex2.find_all("timex2_mention"):
            timex2_mention_id = timex2_mention.attrs['id']

            extend = timex2_mention.find("extent", recursive=False)

            extend_start = int(extend.find("charseq", recursive=False).attrs['start'])
            extend_end = int(extend.find("charseq", recursive=False).attrs['end'])

            timex2_text = timex2_mention.find("charseq").text

            time_mentions.append(TimeX2Mention(
                id=timex2_mention_id,
                element=timex2_id,
                extend_start=extend_start,
                extend_end=extend_end,
                head_start=extend_start,
                head_end=extend_end,
                head_text=timex2_text
            ))

        time = TimeX2(
            id=timex2_id,
            val=timex2_val,
            anchor_val=timex2_anchor_val,
            anchor_dir=timex2_anchor_dir,
            min_extend_char_idx=min([mention.extend_start for mention in time_mentions]),
            max_extend_char_idx=max([mention.extend_end for mention in time_mentions]),
            mentions=time_mentions
        )

        for time_mention in time_mentions:
            time_mention.element = time
            id2time_mention[time_mention.id] = time_mention

        times.append(time)

    id2time = {time.id: time for time in times}

    return times, id2time, id2time_mention


@dataclass
class Entity(Element):
    type: str = None
    subtype: str = None
    class_: str = None

    @overrides
    def to_json(self):
        return {
            'id': self.id,
            'type': self.type,
            'subtype': self.subtype,
            'class': self.class_,
            'min_extend_char_idx': self.min_extend_char_idx,
            'max_extend_char_idx': self.max_extend_char_idx,
            'mentions': [mention.id for mention in self.mentions]
        }

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


@dataclass
class EntityMention(ElementMention):
    type: str = None

    @overrides
    def to_json(self):
        return {
            'id': self.id,
            'element': self.element.id,
            'type': self.type,
            'extend_start': self.extend_start,
            'extend_end': self.extend_end,
            'head_start': self.head_start,
            'head_end': self.head_end,
            'head_text': self.head_text
        }

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


def extract_entities(annotation_text, complete_text):
    soup = BeautifulSoup(annotation_text, 'lxml')
    entities = []
    id2entity_mention = {}
    for entity in soup.find_all("entity"):

        entity_id = entity.attrs['id']
        entity_type = entity.attrs['type']
        entity_subtype = entity.attrs['subtype']
        entity_class = entity.attrs['class'][0]

        entity_min_extend_char_idx = 10000000
        entity_max_extend_char_idx = 0

        entity_mentions = []

        for entity_mention in entity.find_all("entity_mention"):
            entity_mention_id = entity_mention.attrs['id']
            entity_mention_type = entity_mention.attrs['type']

            extend = entity_mention.find("extent", recursive=False)
            head = entity_mention.find("head", recursive=False)

            extend_start = int(extend.find("charseq", recursive=False).attrs['start'])
            extend_end = int(extend.find("charseq", recursive=False).attrs['end'])

            entity_text = entity_mention.find("charseq", recursive=False).text

            if head is not None:
                head_start = int(head.find("charseq", recursive=False).attrs['start'])
                head_end = int(head.find("charseq", recursive=False).attrs['end'])
            else:
                head_start = int(entity_mention.find("charseq", recursive=False).attrs['start'])
                head_end = int(entity_mention.find("charseq", recursive=False).attrs['end'])

            # for k, offset in OFFSET_CORECTION_KEYS.items():
            #
            #     if k in entity_mention_id:
            #         head_start -= offset
            #         head_end -= offset
            #
            #         extend_start -= offset
            #         extend_end -= offset

            ctrl_text = complete_text[head_start: head_end + 1]
            if ctrl_text != entity_text:
                if entity_text == complete_text[head_start - 1: head_end]:
                    head_start -= 1
                    head_end -= 1

            entity_min_extend_char_idx = min(entity_min_extend_char_idx, extend_start)
            entity_max_extend_char_idx = max(entity_max_extend_char_idx, extend_end)

            entity_mentions.append(EntityMention(
                id=entity_mention_id,
                element=entity_id,
                type=entity_mention_type,
                extend_start=extend_start,
                extend_end=extend_end,
                head_start=head_start,
                head_end=head_end,
                head_text=entity_text
            ))

        entity = Entity(
            id=entity_id,
            type=entity_type,
            subtype=entity_subtype,
            class_=entity_class,
            min_extend_char_idx=entity_min_extend_char_idx,
            max_extend_char_idx=entity_max_extend_char_idx,
            mentions=entity_mentions
        )

        for mention in entity_mentions:
            mention.element = entity
            id2entity_mention[mention.id] = mention

        entities.append(entity)

    id2entity = {entity.id: entity for entity in entities}

    return entities, id2entity, id2entity_mention


class TokenMatchingError(Exception):
    pass


def is_mention_in_sentence(mention: ElementMention, sent: Span):
    # Check token boundaries, for mentions end is inclusive, for spans its exclusive
    token_match = mention.head_token_start >= sent.start and mention.head_token_end < sent.end
    char_match = mention.head_start >= sent.start_char and mention.head_end < sent.end_char

    if token_match and char_match:
        return True

    if not (token_match or char_match):
        return False
    return False


def get_elems_in_sentence(elems: list[Element], sent: Span):
    mentions_in_sentence = []

    for elem in elems:
        for mention in elem.mentions:
            if is_mention_in_sentence(mention, sent):
                mentions_in_sentence.append(mention)

    return mentions_in_sentence


def get_relations_in_sentence(sent_ent_mentions: list[ElementMention], relations: list[Relation]):
    mentions_in_sentence = []

    for relation in relations:
        for mention in relation.mentions:
            if mention.arg1 in sent_ent_mentions and mention.arg2 in sent_ent_mentions:
                mentions_in_sentence.append(mention)

    return mentions_in_sentence


def extract_sentences(file_key, entities, times, values, relations, doc):
    sentences = []
    sent: Span
    for sent in doc.sents:
        sent_ent_mentions = get_elems_in_sentence(entities, sent)
        sent_time_mentions = get_elems_in_sentence(times, sent)
        sent_val_mentions = get_elems_in_sentence(values, sent)

        sent_rel_mentions = get_relations_in_sentence(sent_ent_mentions, relations)

        sentences.append(Sentence(id=f"{file_key}-S{sent.start_char}",
                                  start=sent.start_char,
                                  end=sent.end_char - 1,
                                  token_start=sent.start,
                                  token_end=sent.end - 1,
                                  entity_mentions=sent_ent_mentions,
                                  relation_mentions=sent_rel_mentions,
                                  timex2_mentions=sent_time_mentions,
                                  valueMentions=sent_val_mentions,
                                  text=sent.text,
                                  tokens=[t.text for t in sent]))

    return sentences


class Document(BaseModel):
    text: str
    tokens: list[str]
    sentences: list[Sentence]
    entities: list[Entity]

    times: list[TimeX2]
    values: list[Element]
