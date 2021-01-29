import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import timedelta
from tqdm import tqdm
import argparse


def df_to_gdf(df, x, y):
    geometry = [Point(xy) for xy in zip(df[x], df[y])] # create Geometry series with lat / longitude
    df = df.drop([x, y], axis=1)
    gdf = gpd.GeoDataFrame(df, crs=None, geometry=geometry)
    return gdf


def get_nearest_points(k, base_point, candidate_points):
    sindex = candidate_points.sindex # r-tree
    nearest_index = list(sindex.nearest(base_point.geometry.iloc[0].bounds, k))
    nearest_points = candidate_points.iloc[nearest_index]
    return nearest_points


def dist(origin, departure):
    from geopy.distance import geodesic
    geo_dist = geodesic(
        (origin.coords[0][1], origin.coords[0][0]),
        (departure.coords[0][1], departure.coords[0][0])
    ).meters
    return geo_dist


def node_potential(record, venue):
    sigma = 0.001 # 1e-4
    potential_value = np.exp(-(dist(record, venue) / 2*sigma**2))
    return potential_value


def pairwise_potential(nearest_points, edge_df, alpha, beta):
    potential_value = list()
    
    for i in nearest_points.index:
        for edge in edge_df.index:
            if (nearest_points.activity_class[i] == edge_df.activity_class[edge]) and edge_df.geo_distance[edge] < 50:
                potential_value.append(1.0)
            elif nearest_points.activity_class[i] == edge_df.activity_class[edge]:
                potential_value.append(np.exp(-alpha))
            else:
                potential_value.append(np.exp(-beta))
    return potential_value


def pairwise_MRF(node, edges, label, normalize=False, alpha=0.1, beta=0.2):
    node_e = -np.log(node_potential(node.geometry.iloc[0], label.geometry.iloc[0]))
    pairwise_e = sum(-np.log(pairwise_potential(label, edges, alpha, beta)))
    energy = node_e + pairwise_e
    
    if normalize:
        energy = (1 + (energy**2))**-1
    return energy


if __name__ == '__main__':
    # python .\personal_preference.py --D=100 --T=1 --alpha=0.1 --beta=0.2
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--D', required=True, help='')
    parser.add_argument('--T', required=True, help='')
    parser.add_argument('--alpha', required=True, help='')
    parser.add_argument('--beta', required=True, help='')
    args = parser.parse_args()

    DATA_DIR = './data'
    thresh_dist = float(args.D)
    thresh_time = timedelta(hours=(float(args.T)))
    alpha = float(args.alpha)
    beta = float(args.beta)
    # thresh_dist = 100 # ξD: [10, 50, 100, 150, 200]
    # thresh_time = timedelta(hours=1) # ξT: [0.5, 1, 1.5, 2, 2.5, 3]
    # alpha = 0.1 # α: [0.04, 0.06, 0.08, 0.1, 0.12, 0.14]
    # beta = 0.2 # β: [0.15, 0.2, 0.25, 0.3, 0.35]
    
    stop_df = pd.read_csv(DATA_DIR + '/pre_stop_points.csv')
    stop_df.start_time = pd.to_datetime(stop_df.start_time)
    stop_df.end_time = pd.to_datetime(stop_df.end_time)
    stop_gdf = df_to_gdf(stop_df, x='lng', y='lat')

    poi_df = pd.read_csv(DATA_DIR + '/mapped_POIs.csv')
    poi_gdf = df_to_gdf(poi_df, x='lng', y='lat')

    energy_df = pd.DataFrame()

    for uid in tqdm(stop_gdf.uid.unique()):
        user_df = stop_gdf[stop_gdf.uid == uid]
        _user_df = user_df.reset_index(drop=True)
        
        for i in _user_df.index:
            base_point = _user_df.iloc[[i]]
            energy_list = [base_point.id.iloc[0]]
            
            for atv_class in range(4):
                candidate_points = poi_gdf[poi_gdf.activity_class == atv_class]
                nearest_points = get_nearest_points(5, base_point, candidate_points)

                _base_point = [base_point.geometry.iloc[0]] * len(_user_df)
                geo_distance = list(map(dist, _base_point, _user_df.geometry))
                base_time = base_point.start_time.iloc[0]
                timestamp = abs(base_time - _user_df.start_time) % timedelta(days=1)

                _user_df['timestamp'] = timestamp
                _user_df['geo_distance'] = geo_distance
                # edge_df = _user_df[~_user_df.id.isin(base_point.id)] # NR
                # edge_df = _user_df[(_user_df.timestamp < thresh_time) & (~_user_df.id.isin(base_point.id))] # TR
                # edge_df = _user_df[(_user_df.geo_distance < thresh_dist) & (~_user_df.id.isin(base_point.id))] # SR
                edge_df = _user_df[(_user_df.timestamp < thresh_time) & (_user_df.geo_distance < thresh_dist) &
                                (~_user_df.id.isin(base_point.id))] # STR
                energy = list([pairwise_MRF(base_point, edge_df,
                                            label=nearest_points.iloc[[k]],
                                            normalize=False) for k in range(len(nearest_points))])
                
                tmp_df = pd.DataFrame(energy, columns=['energy'])
                argmin = tmp_df.energy.idxmin()
                energy = tmp_df.iloc[argmin].iloc[0]
                energy_list.append(energy)
            energy_df = energy_df.append(pd.DataFrame([energy_list], columns=['id', 'energy_1', 'energy_2', 'energy_3', 'energy_4']))

    energy_df = energy_df.reset_index(drop=True)
    x = (1 + (energy_df[energy_df.columns[1:]]**2))**-1
    x.columns = [0, 1, 2, 3]
    y = stop_gdf['activity_class']
    true_count = 0

    for i in x.index:
        col = x.loc[i].idxmax()
        min_dist = int(col)
        label = int(y[i])
        
        if min_dist == label:
            true_count += 1

    with open(DATA_DIR + '/str_acc.txt', 'a') as file:
        file.write(f'D: {thresh_dist}, T: {thresh_time}, alpha: {alpha}, beta: {beta}, acc: {true_count/len(x) * 100}\n')
    # print(f'D: {thresh_dist}, T: {thresh_time}, alpha: {alpha}, beta: {beta}, acc: {true_count/len(x) * 100}')