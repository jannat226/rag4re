import json
import argparse
import sys
from spacy.lang.en import English
import spacy
import random
from sklearn.model_selection import StratifiedShuffleSplit
random.seed(42)
#valid relations in the format of (subject_type, object_type, predicate from the table )
valid_relations = { #made this into an actual dictionary
    ("anatomical location", "human"): "located in",
    ("anatomical location", "animal"): "located in",
("bacteria", "bacteria"): "interact",
    ("bacteria", "chemical"): "interact",
    ("bacteria", "drug"): "interact",
    ("bacteria", "DDF"): "influence",
    ("bacteria", "gene"): "change expression",
    ("bacteria", "human" ):"located in",
    ("bacteria", "animal" ):"located in",
    ("bacteria", "microbiome" ):"part of",
    ("chemical", "anatomical location" ):"located in",
    ("chemical", "human" ):"located in",
    ("chemical", "animal" ):"located in",
    ("chemical", "chemical"):"interact",
    ("chemical", "chemical"): "part of",
    ("chemical", "microbiome" ):"impact",
    ("chemical", "microbiome" ):"produced by",
    ("chemical", "bacteria" ):"impact",
    ("dietary supplement" ,"bacteria"): "impact",
    ("drug", "bacteria" ):"impact",
    ("food", "bacteria" ):"impact",
    ("chemical", "microbiome" ):"impact",
    ("dietary supplement", "microbiome"): "impact",
    ("drug", "microbiome" ):"impact",
    ("food", "microbiome" ):"impact",
    ("chemical", "DDF" ):"influence",
    ("dietary supplement", "DDF" ):"influence",
    ("food", "DDF" ):"influence",
    ("chemical", "gene" ):"change expression",
    ("dietary supplement", "gene" ):"change expression",
    ("drug", "gene" ):"change expression",
    ("food", "gene" ):"change expression",
    ("chemical", "human" ):"administered",
    ("dietary supplement", "human" ):"administered",
    ("drug", "human" ):"administered",
    ("food", "human" ):"administered",
    ("chemical", "animal" ):"administered",
    ("dietary supplement", "animal" ):"administered",
    ("drug", "animal" ):"administered",
    ("food", "animal" ):"administered",
    ("DDF", "anatomical location" ):"strike",
    ("DDF", "bacteria" ):"change abundance",
    ("DDF", "microbiome" ):"change abundance",
    ("DDF", "chemical" ):"interact",
    ("DDF", "DDF" ):"affect",
    ("DDF", "DDF" ):"is a",
    ("DDF", "human" ):"target",
    ("DDF", "animal" ):"target",
    ("drug", "chemical" ):"interact",
    ("drug", "drug" ):"interact",
    ("drug", "DDF" ):"change effect",
    ("human", "biomedical technique" ):"used by",
    ("animal", "biomedical technique" ):"used by",
    ("microbiome", "biomedical technique" ):"used by",
    ("microbiome", "anatomical location" ):"located in",
    ("microbiome", "human" ):"located in",
    ("microbiome", "animal" ):"located in",
    ("microbiome", "gene" ):"change expression",
    ("microbiome", "DDF" ):"is linked to",
    ("microbiome", "microbiome" ):"compared to"
}

def preprocess_train_data(data):
    processed_data = []
    counter = 0
    counter2 = 0
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe("sentencizer")
    for doc_id, doc_data in data.items():
        metadata = doc_data.get("metadata", {})
        abstract = metadata.get("abstract", "")
        title = metadata.get("title", "")
        relations = doc_data.get("relations", [])
        entities = doc_data.get("entities", [])
        used = []
        title_doc = nlp(title)
        abstract_doc = nlp(abstract)
        """for span in doc.sents:
            print(span.text)
            print(abstract[span.start_char:span.end_char])
            assert span.text == abstract[span.start_char:span.end_char]
        exit()"""#just a test

        # Iteration over relations
        for relation in relations:
            subject_start = relation.get("subject_start_idx")
            subject_end = relation.get("subject_end_idx")
            object_start = relation.get("object_start_idx")
            object_end = relation.get("object_end_idx")
            predicate = relation.get("predicate")
            subject_text = relation.get("subject_text_span")
            object_text = relation.get("object_text_span")
            subject_location = relation.get("subject_location")
            object_location = relation.get("object_location")
            sample = ""
            relative_subject_start= -1
            relative_object_start = -1
            relative_subject_end = -1
            relative_object_end = -1
            if subject_location!=object_location:
                counter2+=1

            if subject_location == "abstract":
                for span in abstract_doc.sents:
                    if span.start_char<= subject_start and span.end_char>=subject_end+1:
                        relative_subject_start = len(sample)+(subject_start-span.start_char)
                        relative_subject_end = len(sample)+(subject_end-span.start_char)
                        sample+=span.text
            elif subject_location == "title":
                for span in title_doc.sents:
                    if span.start_char<=subject_start and span.end_char>=subject_end+1:
                        relative_subject_start = len(sample)+(subject_start-span.start_char)
                        relative_subject_end = len(sample)+(subject_end-span.start_char)
                        sample+=span.text
            if object_location == "abstract":
                for span in abstract_doc.sents:
                    if span.start_char<= object_start and span.end_char>=object_end+1:
                        if sample!=span.text:
                            relative_object_start = len(sample)+(object_start-span.start_char)
                            relative_object_end = len(sample)+(object_end-span.start_char)
                            sample+=span.text
                            if subject_location=="abstract":
                                counter+=1
                        elif sample == span.text:
                            relative_object_start = (object_start-span.start_char)
                            relative_object_end = (object_end-span.start_char)

            elif object_location=="title":
                for span in title_doc.sents:
                    if span.start_char<=object_start and span.end_char>=object_end+1:
                        if sample!=span.text:
                            relative_object_start = len(sample)+(object_start-span.start_char)
                            relative_object_end = len(sample)+(object_end-span.start_char)
                            sample+=span.text
                        elif sample == span.text:
                            relative_object_start = (object_start-span.start_char)
                            relative_object_end = (object_end-span.start_char)
            #print((title+abstract)[relative_subject_start:relative_subject_end])
            # print(subject_text)
            if -1 in [relative_subject_start, relative_subject_end]:
                subject_sample = ""
                if subject_location=="abstract":
                    for span in abstract_doc.sents:
                        if span.start_char<=subject_start and span.end_char>=subject_start:
                            subject_sample+=span.text
                            relative_subject_start = subject_start-span.start_char
                        elif span.start_char<=subject_end+1 and span.end_char>=subject_end+1:
                            prev_len = len(subject_sample)+1
                            subject_sample+=" "+span.text
                            relative_subject_end = prev_len+(subject_end-span.start_char)
                if subject_location=="title":
                    for span in title_doc.sents:
                        if span.start_char<=subject_start and span.end_char>=subject_start:
                            subject_sample+=span.text
                            relative_subject_start = subject_start-span.start_char
                        elif span.start_char<=subject_end+1 and span.end_char>=subject_end+1:
                            prev_len = len(subject_sample)+1
                            subject_sample+=span.text+" "
                            relative_subject_end = prev_len+(subject_end-span.start_char)
                sample = subject_sample + sample
                if -1 not in [relative_object_start, relative_object_end]:
                    relative_object_start+=len(subject_sample)
                    relative_object_end+=len(subject_sample)
            if -1 in [relative_object_start, relative_object_end]:
                object_sample = ""
                prevprev_len = len(sample)
                if object_location=="abstract":
                    for span in abstract_doc.sents:
                        if span.start_char<=object_start and span.end_char>=object_start:
                            object_sample+=span.text
                            relative_object_start = object_start-span.start_char
                        elif span.start_char<=object_end and span.end_char>=object_end:
                            prev_len = len(object_sample)+1
                            object_sample+=" "+span.text
                            relative_object_end = prev_len+(object_end-span.start_char)
                if object_location=="title":
                    for span in title_doc.sents:
                        if span.start_char<=object_start+1 and span.end_char>=object_start+1:
                            object_sample+=span.text
                            relative_object_start = object_start-span.start_char
                        elif span.start_char<=object_end+1 and span.end_char>=object_end+1:
                            prev_len = len(object_sample)+1
                            object_sample+=" "+span.text
                            relative_object_end = prev_len+(object_end-span.start_char)
                sample += object_sample
                relative_object_start+=prevprev_len
                relative_object_end+=prevprev_len

            assert (sample)[relative_subject_start:relative_subject_end + 1] == subject_text
            assert (sample)[relative_object_start:relative_object_end + 1] == object_text
        
            processed_data.append({
                "sample": sample,
                "subject": subject_text,
                "subject_label": relation.get("subject_label"),
                "object": object_text,
                "object_label": relation.get("object_label"),
                "relation": predicate,
                "relative_subject_start": relative_subject_start,
                "relative_subject_end": relative_subject_end,
                "relative_object_start": relative_object_start,
                "relative_object_end": relative_object_end,
                "doc_id": doc_id
            })
            used.append((sample,(relative_subject_start, relative_subject_end), (relative_object_start, relative_object_end)))
        for e in entities: #need this for preprocess_test_data
            for e1 in entities:
                if e != e1:
                    subject_start = e.get("start_idx")
                    subject_end = e.get("end_idx")
                    object_start = e1.get("start_idx")
                    object_end = e1.get("end_idx")
                    subject_location = e.get("location")
                    object_location = e1.get("location")
                    sample = ""
                    relative_subject_start = -1
                    relative_subject_end = -1
                    relative_object_start = -1
                    relative_object_end = -1
                    same_sent = None
                    if subject_location == "abstract":
                        for span in abstract_doc.sents:
                            if span.start_char<= subject_start and span.end_char>=subject_end+1:
                                relative_subject_start = len(sample)+(subject_start-span.start_char)
                                relative_subject_end = len(sample)+(subject_end-span.start_char)
                                sample+=span.text
                            if span.start_char<= object_start and span.end_char>=object_end+1:
                                same_sent = True
                            else:
                                same_sent=False
                                
                    elif subject_location == "title":
                        for span in title_doc.sents:
                            if span.start_char<=subject_start and span.end_char>=subject_end+1:
                                relative_subject_start = len(sample)+(subject_start-span.start_char)
                                relative_subject_end = len(sample)+(subject_end-span.start_char)
                                sample+=span.text
                    if object_location == "abstract":
                        for span in abstract_doc.sents:
                            if span.start_char<= object_start and span.end_char>=object_end+1 and sample!=span.text:
                                relative_object_start = len(sample)+(object_start-span.start_char)
                                relative_object_end = len(sample)+(object_end-span.start_char)
                                sample+=span.text
                            elif span.start_char<= object_start and span.end_char>=object_end+1 and sample == span.text:
                                relative_object_start = (object_start-span.start_char)
                                relative_object_end = (object_end-span.start_char)

                    elif object_location=="title":
                        for span in title_doc.sents:
                            if span.start_char<=object_start and span.end_char>=object_end+1 and sample!=span.text:
                                relative_object_start = len(sample)+(object_start-span.start_char)
                                relative_object_end = len(sample)+(object_end-span.start_char)
                                sample+=span.text
                            elif span.start_char<=object_start and span.end_char>=object_end+1 and sample == span.text:
                                relative_object_start = (object_start-span.start_char)
                                relative_object_end = (object_end-span.start_char)
                    if -1 in [relative_subject_start, relative_subject_end]:
                        subject_sample = ""
                        if subject_location=="abstract":
                            for span in abstract_doc.sents:
                                if span.start_char<=subject_start and span.end_char>=subject_start:
                                    subject_sample+=span.text
                                    relative_subject_start = subject_start-span.start_char
                                elif span.start_char<=subject_end+1 and span.end_char>=subject_end+1:
                                    prev_len = len(subject_sample)+1
                                    subject_sample+=" "+span.text
                                    relative_subject_end = prev_len+(subject_end-span.start_char)
                        if subject_location=="title":
                            for span in title_doc.sents:
                                if span.start_char<=subject_start and span.end_char>=subject_start:
                                    subject_sample+=span.text
                                    relative_subject_start = subject_start-span.start_char
                                elif span.start_char<=subject_end+1 and span.end_char>=subject_end+1:
                                    prev_len = len(subject_sample)+1
                                    subject_sample+=span.text+" "
                                    relative_subject_end = prev_len+(subject_end-span.start_char)
                        sample = subject_sample + sample
                        if -1 not in [relative_object_start, relative_object_end]:
                            relative_object_start+=len(subject_sample)
                            relative_object_end+=len(subject_sample)
                    if -1 in [relative_object_start, relative_object_end]:
                        object_sample = ""
                        prevprev_len = len(sample)
                        if object_location=="abstract":
                            for span in abstract_doc.sents:
                                if span.start_char<=object_start and span.end_char>=object_start:
                                    object_sample+=span.text
                                    relative_object_start = object_start-span.start_char
                                elif span.start_char<=object_end and span.end_char>=object_end:
                                    prev_len = len(object_sample)+1
                                    object_sample+=" "+span.text
                                    relative_object_end = prev_len+(object_end-span.start_char)
                        if object_location=="title":
                            for span in title_doc.sents:
                                if span.start_char<=object_start+1 and span.end_char>=object_start+1:
                                    object_sample+=span.text
                                    relative_object_start = object_start-span.start_char
                                elif span.start_char<=object_end+1 and span.end_char>=object_end+1:
                                    prev_len = len(object_sample)+1
                                    object_sample+=" "+span.text
                                    relative_object_end = prev_len+(object_end-span.start_char)
                        sample += object_sample
                        relative_object_start+=prevprev_len
                        relative_object_end+=prevprev_len


                    # Check if the relation is NONE and if the subject and object types are valid
                    if (sample, (relative_subject_start, relative_subject_end), (relative_object_start, relative_object_end)) not in used:
                        subject_type = e.get("label", "") #FIXED had wrong attribute
                        object_type = e1.get("label", "")

                        # Only include "NONE" relations if they match a valid subject-object pair
                        if (subject_type, object_type) in valid_relations:
                           #  predicate = valid_relations[(subject_type, object_type)] we don't want to overwrite the no relatons
                            processed_data.append({
                                "sample": sample,
                                "subject": e.get("text_span", ""),
                                "subject_label": subject_type,
                                "object": e1.get("text_span", ""),
                                "object_label": object_type,
                                "relation": "NONE", #there's still no relation FOR TEST WE DON'T KNOW THIS
                                "relative_subject_start": relative_subject_start,
                                "relative_subject_end": relative_subject_end,
                                "relative_object_start": relative_object_start,
                                "relative_object_end": relative_object_end,
                                "doc_id":doc_id
                            })

                        else:
                            used.append((sample,(relative_subject_start, relative_subject_end), (relative_object_start, relative_object_end)))
    print("Number of cross sentence (within abstract) entities in relations "+str(counter))
    print("Number of cross sentence (accross title and abstract) entities in relations "+str(counter2))
    return processed_data

def downsample_none(data, fraction=.5): #might want to make this generalized later, but who cares
    tot_rel = 0
    #first we need to figure out how much in total
    dist_frac = {"target": .152,
            "influence":.146,
            "located in":.137,
            "is linked to":.137,
            "affect":.121,
            "impact":.059,
            "used by":0.052,
            "interact":0.039,
            "change abundance":.035,
            "part of":.029,
            "is a": 0.025,
            "change effect":.024,
            "administered":.014,
            "change expression":.012,
            "strike":0.011,
            "produced by":0.005,
            "compared to":0.003
    }
    dist_count = {x:0 for x in dist_frac}

    relations = {key:[] for key in dist_frac}
    new_data = []
    for relation in data:
        if relation["relation"]!="NONE":
            new_data.append(relation)
            dist_count[relation["relation"]]+=1
            tot_rel+=1
        else:
            potential_relation=valid_relations[relation["subject_label"], relation["object_label"]] #need to fix
            relations[potential_relation].append(relation)
    for relation in relations:
        random.shuffle(relations[relation])
    sample_no_rel = [relations[key][:dist_count[key]] for key in dist_count]
    counter = 0

    for sample in sample_no_rel:
        if sample!=[]:
            new_data.extend(sample)
    dist_count_out = {x:0 for x in dist_frac}
    dist_count_out["NONE"] = 0
    for thing in new_data:
        dist_count_out[thing["relation"]]+=1
    print(dist_count_out)

    return new_data



def preprocess_test_data(data):
    processed_data = []
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe("sentencizer")
    
    for doc_id, doc_data in data.items():
        metadata = doc_data.get("metadata", {})
        abstract = metadata.get("abstract", "")
        title = metadata.get("title", "")
        entities = doc_data.get("entities", [])
        title_doc = nlp(title)
        abstract_doc = nlp(abstract)
        
        
        for e in entities: #need this for preprocess_test_data
            for e1 in entities:
                if e != e1:
                    subject_start = e.get("start_idx")
                    subject_end = e.get("end_idx")
                    object_start = e1.get("start_idx")
                    object_end = e1.get("end_idx")
                    subject_location = e.get("location")
                    object_location = e1.get("location")
                    subject_text = e.get("text_span")
                    object_text = e1.get("text_span")

                    sample = ""
                    relative_subject_start = -1
                    relative_subject_end = -1
                    relative_object_start = -1
                    relative_object_end = -1
                    subject_label = e.get("label", "")
                    object_label = e1.get("label", "")
                    if(subject_label, object_label) not in valid_relations:
                        continue
                    if subject_location == "abstract":
                        for span in abstract_doc.sents:
                            if span.start_char<= subject_start and span.end_char>=subject_end+1:
                                relative_subject_start = len(sample)+(subject_start-span.start_char)
                                relative_subject_end = len(sample)+(subject_end-span.start_char)
                                sample+=span.text
                    elif subject_location == "title":
                        for span in title_doc.sents:
                            if span.start_char<=subject_start and span.end_char>=subject_end+1:
                                relative_subject_start = len(sample)+(subject_start-span.start_char)
                                relative_subject_end = len(sample)+(subject_end-span.start_char)
                                sample+=span.text
                    if object_location == "abstract":
                        for span in abstract_doc.sents:
                            if span.start_char<= object_start and span.end_char>=object_end+1 and sample!=span.text:
                                relative_object_start = len(sample)+(object_start-span.start_char)
                                relative_object_end = len(sample)+(object_end-span.start_char)
                                sample+=span.text
                            elif span.start_char<= object_start and span.end_char>=object_end+1 and sample == span.text:
                                relative_object_start = (object_start-span.start_char)
                                relative_object_end = (object_end-span.start_char)

                    elif object_location=="title":
                        for span in title_doc.sents:
                            if span.start_char<=object_start and span.end_char>=object_end+1 and sample!=span.text:
                                relative_object_start = len(sample)+(object_start-span.start_char)
                                relative_object_end = len(sample)+(object_end-span.start_char)
                                sample+=span.text
                            elif span.start_char<=object_start and span.end_char>=object_end+1 and sample == span.text:
                                relative_object_start = (object_start-span.start_char)
                                relative_object_end = (object_end-span.start_char)
                    if -1 in [relative_subject_start, relative_subject_end]:
                        subject_sample = ""
                        if subject_location=="abstract":
                            for span in abstract_doc.sents:
                                if span.start_char<=subject_start and span.end_char>=subject_start:
                                    subject_sample+=span.text
                                    relative_subject_start = subject_start-span.start_char
                                elif span.start_char<=subject_end+1 and span.end_char>=subject_end+1:
                                    prev_len = len(subject_sample)+1
                                    subject_sample+=" "+span.text
                                    relative_subject_end = prev_len+(subject_end-span.start_char)
                        if subject_location=="title":
                            for span in title_doc.sents:
                                if span.start_char<=subject_start and span.end_char>=subject_start:
                                    subject_sample+=span.text
                                    relative_subject_start = subject_start-span.start_char
                                elif span.start_char<=subject_end+1 and span.end_char>=subject_end+1:
                                    prev_len = len(subject_sample)+1
                                    subject_sample+=span.text+" "
                                    relative_subject_end = prev_len+(subject_end-span.start_char)
                        sample = subject_sample + sample
                        if -1 not in [relative_object_start, relative_object_end]:
                            relative_object_start+=len(subject_sample)
                            relative_object_end+=len(subject_sample)
                    if -1 in [relative_object_start, relative_object_end]:
                        object_sample = ""
                        prevprev_len = len(sample)
                        if object_location=="abstract":
                            for span in abstract_doc.sents:
                                if span.start_char<=object_start and span.end_char>=object_start:
                                    object_sample+=span.text
                                    relative_object_start = object_start-span.start_char
                                elif span.start_char<=object_end and span.end_char>=object_end:
                                    prev_len = len(object_sample)+1
                                    object_sample+=" "+span.text
                                    relative_object_end = prev_len+(object_end-span.start_char)
                        if object_location=="title":
                            for span in title_doc.sents:
                                if span.start_char<=object_start+1 and span.end_char>=object_start+1:
                                    object_sample+=span.text
                                    relative_object_start = object_start-span.start_char
                                elif span.start_char<=object_end+1 and span.end_char>=object_end+1:
                                    prev_len = len(object_sample)+1
                                    object_sample+=" "+span.text
                                    relative_object_end = prev_len+(object_end-span.start_char)
                        sample += object_sample
                        relative_object_start+=prevprev_len
                        relative_object_end+=prevprev_len
                    assert (sample)[relative_subject_start:relative_subject_end + 1] == subject_text
                    assert (sample)[relative_object_start:relative_object_end + 1] == object_text


                    processed_data.append({
                        "sample": sample,
                        "subject": e.get("text_span", ""),
                        "subject_label": e.get("label", ""),
                        "object": e1.get("text_span", ""),
                        "object_label": e1.get("label", ""),
                        # "relation": "NONE", #there's still no relation FOR TEST WE DON'T KNOW THIS
                        "relative_subject_start": relative_subject_start,
                        "relative_subject_end": relative_subject_end,
                        "relative_object_start": relative_object_start,
                        "relative_object_end": relative_object_end,
                        "doc_id":doc_id
                    })                       
    return processed_data

    

    
def train_val_split(data):
    rel_counts = {}
    for d in data:
        if d["relation"] not in rel_counts:
            rel_counts[d["relation"]] = 1
        else:
            rel_counts[d["relation"]]+=1
    filtered_data = []
    to_add = []
    for d in data:
        if rel_counts[d["relation"]]>1:
            filtered_data.append(d)
        else:
            to_add.append(d)
    indices = []
    classes = []
    counter = 0
    for d in filtered_data:
        indices.append(counter)
        counter+=1
        classes.append(d["relation"])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    val = None
    train= None
    for i, (train_index, test_index) in enumerate(sss.split(indices, classes)):
        train = [data[x] for x in train_index]
        val = [data[x] for x in test_index]
    train.extend(to_add)
    return train, val

# Save processed data using the 3rd arg 
parser = argparse.ArgumentParser(
                    prog='preprocessing.py',
                    description='Preprocessing for our RE system!!',
                    epilog='~Fin')
output_file = sys.argv[2]
parser.add_argument("--train_out", help="output path to json file for training data")
parser.add_argument("--val_out", help = "output path to json file for val data")
parser.add_argument("--test_out", help = "output path to json file for test data")
parser.add_argument("--train_in", help = "input path training data (json)")
parser.add_argument("--test_in", help = "input path to test data (json)")

args = parser.parse_args()
train_out = args.train_out
val_out = args.val_out
test_out = args.test_out
train_in = args.train_in
test_in = args.test_in
if train_in is not None:
    if train_out is None:
        print("Please provide output path for train data")
        exit(1)
            
    with open(train_in, 'r', encoding='utf-8') as train_in_file:
        data = json.load(train_in_file)
        processed_train_output = preprocess_train_data(data)
        downsampled_processed_output = downsample_none(processed_train_output, .5)
        if val_out is not None:
            with open(train_out, "w", encoding="utf-8") as train_out_file:
                train, val = train_val_split(downsampled_processed_output)
                json.dump(train, train_out_file, indent=4)
                with open(val_out, "w", encoding="utf-8") as val_out_file:
                    json.dump(val, val_out_file, indent=4)
        else:
            with open(train_out, "w", encoding="utf-8") as train_out_file:
                json.dump(downsampled_processed_output, train_out_file, indent=4)
        
if test_in is not None:
    if test_out is None:
        print("Please provide output path for test data")
        exit(1)
    with open(test_in, "r", encoding="utf-8") as test_in_file:
        data = json.load(test_in_file)
        processed_test_output = preprocess_test_data(data)
        with open(test_out, "w", encoding="utf-8") as test_out_file:
            json.dump(processed_test_output, test_out_file, indent=4)

print(f"Done!! :)")



    
    
