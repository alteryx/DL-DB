from keras.layers import Input, Embedding, Dense, Flatten, Reshape, LSTM
from featuretools.entityset.relationship import Relationship
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
import keras
from featuretools.variable_types import Discrete, Numeric, Id, Index
from collections import defaultdict
import pandas as pd
import numpy as np
import copy


def keras_model_from_entityset(es, target,
                               max_values_per_instance,
                               categorical_max_vocab,
                               numeric_output_size=10,
                               numeric_embedding_size=10,
                               categorical_output_size=10,
                               categorical_embedding_size=10,
                               output_layer_size=10):
    '''
    target is variable in es
    '''
    inputs = []
    outputs = []
    entity_deps = get_entity_deps(es, target.entity.id)
    for top_level_entity, deps in entity_deps.items():
        for child in deps:
            if child == top_level_entity and child != target.entity.id:
                continue
            # TODO: see if this is still necessary
            if form_layer_input_name(top_level_entity, child, is_numeric=True) not in max_values_per_instance:
                continue
            layer_inputs, layer_output = build_input_network(
                es[child],
                numeric_output_size=numeric_output_size,
                categorical_output_size=categorical_output_size,
                numeric_embedding_size=numeric_embedding_size,
                categorical_embedding_size=categorical_embedding_size,
                max_values_per_instance=max_values_per_instance[form_layer_input_name(top_level_entity, child, is_numeric=True)],
                categorical_max_vocab=categorical_max_vocab[child],
                name_prefix="{}_".format(top_level_entity),
                ignore=target)
            if layer_output is not None:
                outputs.append(layer_output)
            if layer_inputs is not None:
                inputs.extend(layer_inputs)
    output_from_entities = keras.layers.concatenate(outputs)
    output_layer = Dense(output_layer_size, activation='relu')(output_from_entities)
    main_output = Dense(1, activation='sigmoid', name='target')(output_layer)

    model = Model(inputs=inputs, outputs=main_output)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model


def eid_from_layer_input_name(layer_input_name):
    if layer_input_name.endswith('numeric_vars'):
        eid = layer_input_name.split("_numeric_vars")[0].split("_")[1]
    else:
        eid = layer_input_name.split("_cat_vars")[0].split("_")[1]
    return eid


def form_layer_input_name(parent_eid, eid, is_numeric, var=None):
    if is_numeric:
        return "{}_{}_numeric_vars".format(parent_eid, eid)
    else:
        return "{}_{}_categorical_{}".format(parent_eid, eid, var)


def fit_entityset_using_nn(es, instance_ids, target_variable):
    inputs, cat_mapping = create_inputs_multi(es, target_variable, instance_ids)
    max_values_per_instance = {name: data.shape[1] for name, data in inputs.items()}
    categorical_max_vocab = {name: {v: 1 + len(d) for v, d in data.items()} for name, data in cat_mapping.items()}
    model = keras_model_from_entityset(es, target_variable,
                                       max_values_per_instance=max_values_per_instance,
                                       categorical_max_vocab=categorical_max_vocab)
    df = es.get_instance_data(target_variable.entity.id, instance_ids)
    labels = df[target_variable.id].astype(int).values
    model.fit(inputs, labels, epochs=1, batch_size=4)
    return model


def predict(model, es, target_variable, instance_ids,
            max_values_per_instance, cat_mapping):
    inputs, cat_mapping = create_inputs(es, target_variable, instance_ids,
                           max_values_per_instance=max_values_per_instance,
                           cat_mapping=cat_mapping)
    return model.predict(inputs)


def get_entity_deps(es, target_eid):
    entity_deps = {}
    for e in es.entities:
        _, num_forward = es.find_path(target_eid, e.id, include_num_forward=True)
        if num_forward or target_eid == e.id:
            # is top level
            entity_deps[e.id] = find_all_children(es, e.id)
    return entity_deps


def find_all_children(es, eid):
    children = [eid]
    r_queue = es.get_backward_relationships(eid)
    while r_queue:
        r = r_queue.pop(0)
        child_eid = r.child_variable.entity.id
        # TODO: we can probalby work around this now
        if child_eid in children:
            raise RuntimeError('Diamond graph detected!')
        r_queue += es.get_backward_relationships(child_eid)
        children.append(child_eid)
    return children


def create_inputs_multi(es, target, instance_ids, cat_mapping_by_entity={}):
    '''
    TODO: maybe easier just merge all data together first, and then extract
    out entities/inputs afterwards
    '''
    numeric_vars_by_entity = {entity.id: [v.id for v in entity.variables
                                          if isinstance(v, Numeric)]
                              for entity in es.entities}
    cat_vars_by_entity = {entity.id: [v.id for v in entity.variables
                                      if isinstance(v, Discrete) and
                                      not isinstance(v, Id)]
                          for entity in es.entities}
    es_with_categorical_values, cat_mapping_by_entity = map_categoricals(es, cat_mapping_by_entity)

    eframes = es_with_categorical_values.get_pandas_data_slice([e.id for e in es.entities],
                                                               target.entity.id,
                                                               instance_ids)

    full_inputs = defaultdict(list)
    # instance_id_col = es[target.entity.id].index
    for parent_eid, child_eframes in eframes.items():
        top_level_instances = eframes[parent_eid][parent_eid]
        top_level_instances, instance_id_col = add_instance_id_col_to_frame(
            es_with_categorical_values, target.entity.id, parent_eid, top_level_instances,
            instance_ids)
        if instance_id_col not in top_level_instances:
            import pdb; pdb.set_trace()
        parent_index = es[parent_eid].index
        for eid, eframe in child_eframes.items():
            if eid == parent_eid and eid != target.entity.id:
                continue
            numeric_vars = numeric_vars_by_entity[eid]
            cat_vars = cat_vars_by_entity[eid]
            if numeric_vars or cat_vars:
                numeric_input_name = form_layer_input_name(parent_eid, eid, is_numeric=True)
                if parent_eid == target.entity.id == eid:
                    # 1 per instance
                    if len(numeric_vars):
                        numeric_inputs = eframe[numeric_vars]
                        # TODO: do we need a list?
                        full_inputs[numeric_input_name] = numeric_inputs.values
                    for var in cat_vars:
                        input_name = form_layer_input_name(parent_eid, eid, is_numeric=False, var=var)
                        full_inputs[input_name] = eframe[[var]].values
                else:
                    parent_to_child = es.find_path(parent_eid, eid)
                    link_var = Relationship._get_link_variable_name(parent_to_child)
                    if link_var not in eframe:
                        import pdb; pdb.set_trace()
                    if parent_index not in top_level_instances:
                        # TODO: this might be wrong
                        parent_index = instance_id_col
                        top_level_link_cols = [instance_id_col]
                    elif parent_index == instance_id_col:
                        top_level_link_cols = [parent_index]
                    else:
                        top_level_link_cols = [parent_index, instance_id_col]
                    linked_frame = eframe.merge(top_level_instances[top_level_link_cols],
                                                left_on=link_var, right_on=parent_index)
                    for _, group in linked_frame.groupby(instance_id_col):
                        if len(numeric_vars):
                            full_inputs[numeric_input_name].append(group[numeric_vars].values)
                        for var in cat_vars:
                            input_name = form_layer_input_name(parent_eid, eid, is_numeric=False, var=var)
                            full_inputs[input_name].append(group[var].values)
    for name, data in full_inputs.items():
        # TODO: each set of sequences should probably be its own network with its own padding
        #pad_value = find_pad_value(data, name)
        if isinstance(data, list):
            full_inputs[name] = pad_sequences(data, padding='pre')
    return full_inputs, cat_mapping_by_entity


def add_instance_id_col_to_frame(es, target_eid, toplevel_eid, frame, instance_ids):
    id_col = es[target_eid].index

    if target_eid == toplevel_eid:
        new_id_col = "{}.{}".format(target_eid, id_col)
        frame = frame.rename(columns={id_col: new_id_col})
        return frame, new_id_col

    id_col = es[target_eid].index

    path = es.find_path(toplevel_eid, target_eid)

    prev_entity = toplevel_eid
    col_map = {c: c for c in frame.columns}
    original_columns = list(frame.columns)
    for r in path:
        if r.child_entity.id == prev_entity:
            left_merge_var = col_map[r.child_variable.id]
            other_e = r.parent_entity
            right_var = r.parent_variable.id
        else:
            left_merge_var = col_map[r.parent_variable.id]
            other_e = r.child_entity
            right_var = r.child_variable.id

        prev_entity = other_e.id

        new_frame_index_cols = [v.id for v in other_e.variables
                                if isinstance(v, (Index, Id))]
        new_frame = other_e.df[new_frame_index_cols]

        # only take relevant instances from target
        if other_e.id == target_eid:
            new_frame = new_frame.loc[instance_ids]

        col_map = {c: '{}.{}'.format(other_e.id, c)
                   for c in new_frame_index_cols}
        new_frame = new_frame.rename(columns=col_map)
        frame = frame.merge(new_frame, left_on=left_merge_var,
                            right_on=col_map[right_var], how='inner')
    instance_id_col = col_map[id_col]
    frame = frame[original_columns + [instance_id_col]]
    return frame, instance_id_col


def map_categoricals_to_integers(input_series, mapping=None):
    vals = input_series.unique()
    if mapping is None:
        mapping = {v: k + 1 for k, v in enumerate(vals)}
    numeric = input_series.replace(mapping)
    return numeric, mapping


def create_inputs(es, target, instance_ids, max_values_per_instance={}, cat_mapping_by_entity=None):
    numeric_vars_by_entity = {entity.id: [v.id for v in entity.variables
                                          if isinstance(v, Numeric)]
                              for entity in es.entities}
    cat_vars_by_entity = {entity.id: [v.id for v in entity.variables
                                      if isinstance(v, Discrete)
                                      and not isinstance(v, Id)]
                          for entity in es.entities}
    full_inputs = defaultdict(list)

    es_with_categorical_values, cat_mapping_by_entity = map_categoricals(es, cat_mapping_by_entity)

    for i in instance_ids:
        # TODO: obviously try to do this more than 1 at a time
        eframes = es_with_categorical_values.get_pandas_data_slice([e.id for e in es.entities], target.entity.id,
                                                                   [i])
        for parent_eid, child_eframes in eframes.items():
            for eid, eframe in child_eframes.items():
                if eid == parent_eid and eid != target.entity.id:
                    continue
                if numeric_vars_by_entity[eid]:
                    input_name = form_layer_input_name(parent_eid, eid, is_numeric=True)
                    numeric_inputs = eframe[numeric_vars_by_entity[eid]]
                    full_inputs[input_name].append(numeric_inputs.values)
                for var in cat_vars_by_entity[eid]:
                    input_name = form_layer_input_name(parent_eid, eid, is_numeric=False, var=var)
                    cat_inputs = eframe[var]
                    full_inputs[input_name].append(cat_inputs.values)
    for name, data in full_inputs.items():
        # TODO: each set of sequences should probably be its own network with its own padding
        #pad_value = find_pad_value(data, name)
        full_inputs[name] = pad_sequences(data,
                                              maxlen=max_values_per_instance.get(name, None),
                                              padding='pre')
    return full_inputs, cat_mapping_by_entity


def map_categoricals(es, cat_mapping_by_entity=None):
    if cat_mapping_by_entity is None:
        cat_mapping_by_entity = {}
    new_es = copy.deepcopy(es)
    for e in new_es.entities:
        cat_mapping = cat_mapping_by_entity.get(e.id, {})
        new_df = e.df.copy()
        for v in e.variables:
            if isinstance(v, Discrete) and not isinstance(v, Id):
                numeric_series, new_mapping = map_categoricals_to_integers(e.df[v.id],
                                                    mapping=cat_mapping.get(v.id, None))
                cat_mapping[v.id] = new_mapping
                new_df[v.id] = numeric_series
        cat_mapping_by_entity[e.id] = cat_mapping
        new_es[e.id].df = new_df
    return new_es, cat_mapping_by_entity


def build_input_network(entity, numeric_output_size,
                        categorical_output_size,
                        numeric_embedding_size,
                        categorical_embedding_size,
                        max_values_per_instance,
                        categorical_max_vocab,
                        name_prefix=None,
                        ignore=None):
    # TODO: index vars can provide Count() as input

    name_prefix = name_prefix or ""
    numeric_input_vars = [v for v in entity.variables
                          if isinstance(v, Numeric) and
                          not v == ignore]
    numeric_nn = None
    if len(numeric_input_vars):
        numeric_input = Input(shape=(max_values_per_instance, len(numeric_input_vars)),
                              dtype='float32',
                              name=name_prefix + "{}_numeric_vars".format(entity.id))
        # TODO: include attention layer instead
        num_forward = LSTM(numeric_embedding_size,
                             return_sequences=False,
                             dropout=0.2,
                             recurrent_dropout=0.2,
                             input_shape=(max_values_per_instance, len(numeric_input_vars)))(numeric_input)
        num_backward = LSTM(numeric_embedding_size,
                            return_sequences=False,
                            dropout=0.2,
                            recurrent_dropout=0.2,
                            go_backwards=True,
                            input_shape=(max_values_per_instance, len(numeric_input_vars)))(numeric_input)
        num_embedding = keras.layers.concatenate([num_forward, num_backward])
        numeric_nn = add_layers(num_embedding, numeric_output_size)
        #numeric_nn = Reshape((numeric_output_size * len(numeric_input_vars),))(numeric_nn)
    cat_input_vars = [v for v in entity.variables
                      if isinstance(v, Discrete) and
                      not isinstance(v, Id) and
                      not v == ignore]

    cat_embedding_nn = []
    cat_inputs = []
    for v in cat_input_vars:
        cat_input = Input(shape=(max_values_per_instance,),
                          dtype='int32',
                          name=name_prefix + "{}_categorical_{}".format(entity.id, v.id))
        cat_inputs.append(cat_input)
        cat_embedding = Embedding(output_dim=categorical_embedding_size,
                                  input_dim=categorical_max_vocab[v.id],
                                  input_length=max_values_per_instance)(cat_input)
        cat_embedding_nn.append(cat_embedding)

    cat_nn = None
    if len(cat_embedding_nn):
        if len(cat_embedding_nn) > 1:
            cat_embedding_nn = keras.layers.concatenate(cat_embedding_nn)
        else:
            cat_embedding_nn  = cat_embedding_nn[0]
        cat_forward = LSTM(categorical_embedding_size,
                           return_sequences=False,
                           dropout=0.2,
                           recurrent_dropout=0.2,
                           input_shape=(categorical_embedding_size, len(cat_input_vars)))(cat_embedding_nn)
        cat_nn = add_layers(cat_forward, categorical_output_size)

    if numeric_nn is not None and cat_nn is not None:
        return [numeric_input] + cat_inputs, keras.layers.concatenate([numeric_nn, cat_nn])
    elif numeric_nn is not None:
        return [numeric_input], numeric_nn
    elif cat_nn is not None:
        return cat_inputs, cat_nn


def add_layers(input_layer, output_size):
    x1 = Dense(10, activation='relu')(input_layer)
    x2 = Dense(output_size, activation='relu')(x1)
    return x2

