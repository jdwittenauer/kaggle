import pandas as pd
import numpy as np
import random
import operator


def apk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    return np.mean([apk(a, p ,k) for a, p in zip(actual, predicted)])


def make_key(items):
    return '_'.join([str(i) for i in items])


def generate_exact_matches(row, match_cols, groups):
    index = tuple([row[t] for t in match_cols])
    try:
        group = groups.get_group(index)
    except Exception:
        return []

    clus = list(set(group.hotel_cluster))

    return clus


def f5(seq, idfun=None):
    if idfun is None:
        def idfun(x): return x

    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)

    return result


def generate_submission(data_dir, preds, test):
    write_p = [' '.join([str(l) for l in p]) for p in preds]
    write_frame = ['{0},{1}'.format(test['id'].iloc[i], write_p[i]) for i in range(len(preds))]
    write_frame = ['id,hotel_cluster'] + write_frame

    with open(data_dir + 'predictions.csv', 'w+') as f:
        f.write('\n'.join(write_frame))


print('Loading data sets...')
data_dir = '/home/data/expedia/'

destinations = pd.read_csv(data_dir + 'destinations.csv')

train = pd.read_csv(data_dir + 'train.csv',
                    usecols=['date_time', 'user_location_country', 'user_location_region', 'user_location_city',
                             'user_id', 'is_booking', 'orig_destination_distance',
                             'hotel_cluster', 'srch_ci', 'srch_co', 'srch_destination_id',
                             'hotel_continent', 'hotel_country', 'hotel_market'],
                    dtype={'date_time': np.str_, 'user_location_country': np.int8,
                           'user_location_region': np.int8, 'user_location_city': np.int8,
                           'user_id': np.int32, 'is_booking': np.int8,
                           'orig_destination_distance': np.float64,
                           'hotel_cluster': np.int8,
                           'srch_ci': np.str_, 'srch_co': np.str_,
                           'srch_destination_id': np.int32,
                           'hotel_continent': np.int8,
                           'hotel_country': np.int8,
                           'hotel_market': np.int8})

test = pd.read_csv(data_dir + 'test.csv',
                   usecols=['id', 'date_time', 'user_location_country', 'user_location_region',
                            'user_location_city',
                            'user_id', 'orig_destination_distance',
                            'srch_ci', 'srch_co', 'srch_destination_id',
                            'hotel_continent', 'hotel_country', 'hotel_market'],
                   dtype={'id': np.int32, 'date_time': np.str_, 'user_location_country': np.int8,
                          'user_location_region': np.int8, 'user_location_city': np.int8,
                          'user_id': np.int32,
                          'orig_destination_distance': np.float64, 'srch_ci': np.str_, 'srch_co': np.str_,
                          'srch_destination_id': np.int32,
                          'hotel_continent': np.int8,
                          'hotel_country': np.int8,
                          'hotel_market': np.int8})

print('Generating first set of predictions...')

# add year and month features to the training data
train['date_time'] = pd.to_datetime(train['date_time'])
train['year'] = train['date_time'].dt.year
train['month'] = train['date_time'].dt.month

# generate a list of randomly selected unique user ids
unique_users = train.user_id.unique()
sel_user_ids = [unique_users[i] for i in sorted(random.sample(range(len(unique_users)), 10000))]
sel_train = train[train.user_id.isin(sel_user_ids)]

# create sampled training and validation data sets
t1 = sel_train[((sel_train.year == 2012) | (sel_train.year == 2013))]
t2 = sel_train[(sel_train.year == 2014)]
t2 = t2[t2.is_booking == True]

# skip sampling and use full data set
# t1 = train
# t2 = test

# identify the most common clusters
most_common_clusters = list(train.hotel_cluster.value_counts().head().index)

# match clusters to search destination
match_cols = ['srch_destination_id']
cluster_cols = match_cols + ['hotel_cluster']
groups = t1.groupby(cluster_cols)

top_clusters = {}
for name, group in groups:
    clicks = len(group.is_booking[group.is_booking == False])
    bookings = len(group.is_booking[group.is_booking == True])

    score = bookings + .15 * clicks

    clus_name = make_key(name[:len(match_cols)])

    if clus_name not in top_clusters:
        top_clusters[clus_name] = {}

    top_clusters[clus_name][name[-1]] = score

# find the top 5 for each search destination
cluster_dict = {}
for n in top_clusters:
    tc = top_clusters[n]
    top = [l[0] for l in sorted(tc.items(), key=operator.itemgetter(1), reverse=True)[:5]]
    cluster_dict[n] = top

# generate predictions based on the top clusters per search destination
preds = []
for index, row in t2.iterrows():
    key = make_key([row[m] for m in match_cols])

    if key in cluster_dict:
        preds.append(cluster_dict[key])
    else:
        preds.append([])

print('Generating second set of predictions...')

# use data leak to match users between train and test data
match_cols = ['user_location_country', 'user_location_region', 'user_location_city', 'hotel_market', 'orig_destination_distance']
groups = t1.groupby(match_cols)

exact_matches = []
for i in range(t2.shape[0]):
    exact_matches.append(generate_exact_matches(t2.iloc[i], match_cols, groups))

# generate predictions
full_preds = [f5(exact_matches[p] + preds[p] + most_common_clusters)[:5] for p in range(len(preds))]

# evaluate the accuracy of this solution
print('Score = ' + str(mapk([[l] for l in t2['hotel_cluster']], full_preds, k=5)))

# print('Writing submission file...')
# generate_submission(data_dir, full_preds, t2)

print('Script complete.')
