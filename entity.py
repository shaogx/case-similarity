def get_entity(tag_seq, char_seq):
    Location = get_Location_entity(tag_seq, char_seq)
    Time = get_TIME_entity(tag_seq, char_seq)
    Means = get_Means_entity(tag_seq, char_seq)
    Thing = get_Thing_entity(tag_seq, char_seq)

    Location = list(set(Location))
    Time = list(set(Time))
    Means = list(set(Means))
    Thing = list(set(Thing))

    return Location, Time, Means, Thing

def get_TIME_entity(tag_seq, char_seq):
    length = len(char_seq)
    TIME = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-Time':
            if 'time' in locals().keys():
                TIME.append(time)
                del time
            time = char
            if i+1 == length:
                TIME.append(time)
        if tag == 'I-Time':
            if 'time' not in locals().keys():
                time = ''
            time += char
            if i+1 == length:
                TIME.append(time)
        if tag not in ['I-Time', 'B-Time']:
            if 'time' in locals().keys():
                TIME.append(time)
                del time
            continue
    return TIME

def get_PER_entity(tag_seq, char_seq):
    length = len(char_seq)
    PER = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-PER':
            if 'per' in locals().keys():
                PER.append(per)
                del per
            per = char
            if i+1 == length:
                PER.append(per)
        if tag == 'I-PER':
            per += char
            if i+1 == length:
                PER.append(per)
        if tag not in ['I-PER', 'B-PER']:
            if 'per' in locals().keys():
                PER.append(per)
                del per
            continue
    return PER

def get_Location_entity(tag_seq, char_seq):
    length = len(char_seq)
    LOC = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-Location':
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            loc = char
            if i+1 == length:
                LOC.append(loc)
        if tag == 'I-Location':
            if 'loc' not in locals().keys():
                loc = ''
            loc += char
            if i+1 == length:
                LOC.append(loc)
        if tag not in ['I-Location', 'B-Location']:
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            continue
    return LOC

def get_Means_entity(tag_seq, char_seq):
    length = len(char_seq)
    Means = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-Means':
            if 'means' in locals().keys():
                Means.append(means)
                del means
            means = char
            if i+1 == length:
                Means.append(means)
        if tag == 'I-Means':
            if 'means' not in locals().keys():
                means = ''
            means += char
            if i+1 == length:
                Means.append(means)
        if tag not in ['I-Means', 'B-Means']:
            if 'means' in locals().keys():
                Means.append(means)
                del means
            continue
    return Means

def get_Thing_entity(tag_seq, char_seq):
    length = len(char_seq)
    Thing = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-Thing':
            if 'thing' in locals().keys():
                Thing.append(thing)
                del thing
            thing = char
            if i+1 == length:
                Thing.append(thing)
        if tag == 'I-Thing':
            if 'thing' not in locals().keys():
                thing = ''
            thing += char
            if i+1 == length:
                Thing.append(thing)
        if tag not in ['I-Thing', 'B-Thing']:
            if 'thing' in locals().keys():
                Thing.append(thing)
                del thing
            continue
    return Thing


def get_LOC_entity(tag_seq, char_seq):
    length = len(char_seq)
    LOC = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-LOC':
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            loc = char
            if i+1 == length:
                LOC.append(loc)
        if tag == 'I-LOC':
            loc += char
            if i+1 == length:
                LOC.append(loc)
        if tag not in ['I-LOC', 'B-LOC']:
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            continue
    return LOC


def get_ORG_entity(tag_seq, char_seq):
    length = len(char_seq)
    ORG = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-ORG':
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            org = char
            if i+1 == length:
                ORG.append(org)
        if tag == 'I-ORG':
            org += char
            if i+1 == length:
                ORG.append(org)
        if tag not in ['I-ORG', 'B-ORG']:
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            continue
    return ORG

