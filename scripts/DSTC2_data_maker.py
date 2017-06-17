import os
import numpy as np
import pandas as pd
import argparse
import json
import copy

def read_json(path):
    return json.load(open(path, 'r'))

def dump_json(path, obj):
    with open(path, 'w') as fn:
        json.dump(obj, fn)

def delete_duplicates(json_list):
    """
    Left only items with unique nl values
    in list of jsons which is looks like this:
    json_list[
             {'nl': 'THIS_SENT_IS_THE_KEY', ...},
             {...},
        ...]
    This method doesn't change json_list;
    Compare only nl values;
    
    """
    nls = []
    res = []
    for item in json_list:
        if item['nl'] not in nls:
            nls.append(item['nl'])
            res.append(item)
    print('unique items: {}'.format(len(res)))
    return res

def process_log(log, who='usr'):
    dialog = {}
    turns_arr = []
    for turn in log['turns']:
        acts_for_item = []
        tmp = {}
        tmp['slots'] = []
        tmp['values'] = []
        tmp['acts'] = []
        tmp['set_acts'] = []
        
        if who == 'usr':
            tmp['nl'] = turn['transcription']
            dialog[turn['turn-index']] = tmp['nl']
            semantics = turn['semantics']['json']
        elif who == 'agt':
            tmp['nl'] = turn['output']['transcript']
            dialog[turn['turn-index']] = tmp['nl']
            semantics = turn['output']['dialog-acts']
        else:
            raise Exception('Wrong value of "who" param!')
        
        # every sentence may have more then 1 slot;
        for item in semantics:
            act = item['act']
            if act:
                tmp['acts'].append(act)
                
            if item['slots']:
                slots = item['slots'][0] # because of such 
                                         #representation: [[]]
                if act != 'request':
                    tmp['slots'].append(slots[0])
                    tmp['values'].append(slots[1])
                else: # because in this case slots[0] == 'slot'
                    tmp['slots'].append(slots[1])
                    tmp['values'].append("NONE")
            
        tmp['set_acts'] = frozenset(tmp['acts'])
        turns_arr.append(tmp)
        
    return turns_arr, dialog


def make_placeholders(df):
    

    req_parse_vals = {
    'phone':['phone number', 'phone', 'number', 'telephone'],
    'addr':['address', 'addre', 'addrss', 'adddress'],
    'pricerange':['price range', 'price ran', 'price of food', 'price code', 'prices', 'price', 'cost'],
    'area':['area', 'part of town', 'location'],
    'food':['food', 'serves'],
    'postcode': ['postcode', 'post code', 'zip code', 'postal code'],
    'name':['name'] }
    
    pandas_df = copy.deepcopy(df)
    for index,row in pandas_df.iterrows():
        for slot,val,act in zip(row['slots'], row['values'], row['acts']):
            if ((slot == 'addr') or (slot == 'postcode')) and act!='request':
                pandas_df.loc[index,'nl'] = row['nl'].replace(val.title(), "${}$".format(slot))
                if slot == 'postcode':
                    pandas_df.loc[index,'nl'] = row['nl'][:-7] # да, тут полно костылей, но иначе никак(
            elif slot!='this':
                if ((val!='NONE')&(val!='dontcare')):
                    pandas_df.loc[index,'nl'] = row['nl'].replace(val, "${}$".format(slot))
                elif (val == 'dontcare'):
                    for assump in ['any', 'you dont care']:
                        pandas_df.loc[index,'nl'] = row['nl'].replace(assump, "${}$".format(slot))
                elif (slot!='signature'):
                    # request case -- we don't know the exactly value of slot;
                    if slot == 'name':
                        print('nl name: ', row['nl'])
                    for v in req_parse_vals[slot]:
                        if v in row['nl']:
                            pandas_df.set_value(index,'nl', row['nl'].replace(v, "${}$".format(slot)))
                            break
                
    pandas_df['nl'] = [' '.join(nl.replace('noise', '').split()) for nl in pandas_df['nl']]
    pandas_df['nl'] = [' '.join(nl.replace('breath', '').split()) for nl in pandas_df['nl']]
    pandas_df['nl'] = [' '.join(nl.replace('unintelligible', '').split()) for nl in pandas_df['nl']]
    return pandas_df


def make_right_columns(df,who, wrong_acts = ['repeat', 'ack', 'restart']):
    """
    drop all rows which have even one of actions in wrong_acts;
    make 3 columns for all actions and 3 columns for all slots;
    drop duplicates;
    it doesn't change original df;
    remove 'noise' word from each nl
    
    """
    def replace_this_slot(df):
        def delete_unnecessary_acts(row, index):
            # если нет для второстепенных действий слотов, то убираем эти действия
            if (row['act2']!='') and (row['slot2']==''):
                df.set_value(index, 'act2','')
            if (row['act3']!='') and (row['slot3']==''):
                df.set_value(index, 'act3','')
                
        if who == 'usr':
            for index, row in df.iterrows():
                delete_unnecessary_acts(row, index)
                for i in range(1,4):
                    if row['act'+str(i)] == 'deny':
                        df.set_value(index,'act'+str(i),'negate')
                        
                    # если 'this' слоту соотвествует экшон 'inform', то заменяем его на 'dontcare':
                    if (row['act'+str(i)]=='inform') and (row['slot'+str(i)]=='this'):
                        df.set_value(index,'act'+str(i), 'dontcare')
                        df.set_value(index,'slot'+str(i), '')
                        
                    elif row['slot'+str(i)]=='this':
                        if row['slot'+str((i+1)%4)]!='':
                            # если следующий за this слот есть, то ставим его на место this
                            # и соответствующий экшон тоже ставим на экшон this
                            df.set_value(index,'slot'+str(i), row['slot'+str((i+1)%4)])
                            df.set_value(index,'slot'+str((i+1)%4), '')
                            df.set_value(index,'act'+str(i), row['act'+str((i+1)%4)])
                            df.set_value(index,'act'+str((i+1)%4), '')
                        else:
                            df.set_value(index,'slot'+str(i), '')
        elif who == 'agt':
            for index, row in df.iterrows():
                delete_unnecessary_acts(row, index)
                if row['act2'] == 'canthelp.exception':
                    df.set_value(index,'act2', '')
                    df.set_value(index,'act3', '')
                    df.set_value(index,'slot2', '')
                    df.set_value(index,'slot3', '')
            
    new_df = copy.deepcopy(df)
    print('shape before:', new_df.shape)
    new_df['act1'] = [act[0] if act else '' for act in df['acts']]
    new_df['act2'] = [act[1] if len(act)>=2 else '' for act in df['acts']]
    new_df['act3'] = [act[2] if len(act)>=3 else '' for act in df['acts']]
    new_df['slot1'] = [slot[0] if slot else '' for slot in df['slots']]
    new_df['slot2'] = [slot[1] if len(slot)>=2 else '' for slot in df['slots']]
    new_df['slot3'] = [slot[2] if len(slot)>=3 else '' for slot in df['slots']]
    replace_this_slot(new_df)
    
    new_df.drop(new_df[new_df.act1 == ''].index, inplace=True) ## delete rows without any info
    new_df.drop(['acts', 'slots', 'set_acts', 'values'], axis=1, inplace=True)
    for act in wrong_acts:
        new_df.drop(new_df[(new_df.act1 == act)|(new_df.act2 == act)|(new_df.act3 == act)].index, inplace=True)
    new_df['nl'] = [nl.replace('$pricerange$ly', '$pricerange$') for nl in new_df['nl']]
    new_df.drop(new_df[new_df.nl == ''].index, inplace=True)
    new_df.drop(new_df[new_df.act1 == 'confirm-domain'].index, inplace=True)
    new_df.drop(new_df[(new_df.slot1 == 'signature')|
                       (new_df.slot2 == 'signature')|
                       (new_df.slot3 == 'signature')].index, inplace=True)
    if who == 'usr':
        new_df.drop(new_df[new_df.act1 == 'reqmore'].index, inplace=True)
    
    new_df.drop_duplicates(inplace=True)
    print('shape after:', new_df.shape)
    return new_df

def canthelp_implconf_edit(df):
    """
    add necessary slots to table using ontology file
    only to columns with canthelp action1
    
    """
    ontology = read_json(ontology_path)
    for index, row in df.iterrows():
        
        if row['act1'] == 'impl-conf':
            df.set_value(index, 'act1', 'inform')
        if row['act2'] == 'impl-conf':
            df.set_value(index, 'act2', 'inform')
        if row['act3'] == 'impl-conf':
            df.set_value(index, 'act3', 'inform')
        
        vacant = 2
        if row['act1'] == 'canthelp':
            for price in ontology['informable']['pricerange']:
                if row['nl'].find(price)>0:
                    df.set_value(index, 'nl', row['nl'].replace(price, '$pricerange$'))
                    df.set_value(index, 'slot'+str(vacant), 'pricerange')
                    df.set_value(index, 'act'+str(vacant), 'canthelp')
                    vacant+=1
                    break
            for area in ontology['informable']['area']:
                if row['nl'].find(area)>0:
                    df.set_value(index, 'nl', row['nl'].replace(area, '$area$'))
                    df.set_value(index, 'slot'+str(vacant), 'area')
                    df.set_value(index, 'act'+str(vacant), 'canthelp')
                    vacant+=1
                    break
            if vacant!=4:
                for food in ontology['informable']['food']:
                    if row['nl'].find(food)>0:
                        df.set_value(index, 'nl', row['nl'].replace(food, '$food$'))
                        df.set_value(index, 'slot'+str(vacant), 'food')
                        df.set_value(index, 'act'+str(vacant), 'canthelp')
                        break
    df.drop_duplicates(inplace=True)
    print('Now shape:', df.shape)


def hello_reqalts_thankyou_dontcare_edit(df):
    '''
    replace 'hello' with 'inform' when necessary
    
    '''
    for index, row in df.iterrows():
        if row['act1'] == 'thankyou':
            df.set_value(index, 'act1', 'bye')
            continue
            
        if (row.act1 == 'hello')|(row.act2 == 'hello')|(row.act3 == 'hello'):
            if row.slot1!='':
                df.set_value(index, 'act1', 'inform')
                
        if (row.act1 == 'reqalts') or (row.act1 == 'dontcare') or (row.act2 == 'dontcare'):
            slots = [row['slot'+str(i)] for i in range(1,4) if row['slot'+str(i)]!='']
            if (row.act1 == 'reqalts') and len(slots)>0:
                df.set_value(index, 'act1', 'inform')
            if ((row.act1 == 'dontcare') or (row.act2 == 'dontcare')) and len(slots)>0:
                df.drop(index, axis=0, inplace=True)


############################################################
#############			MAIN			####################
############################################################

parser = argparse.ArgumentParser()
parser.add_argument("--mfolder", help="main folder with all files", default='../data/dstc2_all/original_data/')
parser.add_argument("--flist", help="list of file names to parse in main folder", default='../data/dstc2_all/original_data/scripts/config/dstc2_train.flist')
parser.add_argument("--out_folder", help="file name to output row dialogs, agt & usr tables", default='.')
parser.add_argument("--ontology_path", help="path for ontology file", default='../data/dstc2_all/original_data/scripts/config/ontology_dstc2.json')

args = parser.parse_args()

src_path = args.mfolder
train = args.flist
out_folder = args.out_folder
ontology_path = args.ontology_path

trn_flist = np.array(pd.read_csv(train, sep='\n', header=None)).flatten()

diaacts_usr = []
diaacts_agt = []


with open(os.path.join(out_folder, "dialog_raw.txt"), 'w') as diatext:
    for i,f in enumerate(trn_flist):
        label_path = os.path.join(src_path, f, 'label.json')
        log_path = os.path.join(src_path, f, 'log.json')

        label = read_json(label_path)
        log = read_json(log_path)
        
        acts_agt, dialog_agt = process_log(log, 'agt')
        acts_usr, dialog_usr = process_log(label, 'usr')
        
        diaacts_usr.extend(acts_usr)
        diaacts_agt.extend(acts_agt)
        
        diatext.write(">{}\n".format(i))
        for j in range(len(dialog_agt)):
            diatext.write("{}\n".format(dialog_agt[j]))
            diatext.write("{}\n".format(dialog_usr[j]))

diaacts_usr = delete_duplicates(diaacts_usr)
diaacts_agt = delete_duplicates(diaacts_agt)

usr_df = pd.DataFrame(diaacts_usr)
agt_df = pd.DataFrame(diaacts_agt)

agt_df = make_placeholders(agt_df)
usr_df = make_placeholders(usr_df)

usr_df = make_right_columns(usr_df, 'usr')
agt_df = make_right_columns(agt_df, 'agt')

canthelp_implconf_edit(agt_df)
hello_reqalts_thankyou_dontcare_edit(usr_df)

usr_df.to_csv(os.path.join(out_folder, "usr_df_final.csv"), index=False)
agt_df.to_csv(os.path.join(out_folder, "agt_df_final.csv"), index=False)



