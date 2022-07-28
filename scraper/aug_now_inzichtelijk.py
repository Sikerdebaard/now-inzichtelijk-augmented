import requests
import tempfile
import git
import json

import numpy as np
import geopandas as gpd
import pandas as pd

from pathlib import Path
from tqdm.auto import tqdm
from textdistance import levenshtein


outdir = Path('data')
outdir.mkdir(exist_ok=True)


def cleanup():
    for f in outdir.glob('*'):
        if f.name == 'latest_commit_now_inzichtelijk':
            continue
        print(f'CLEANUP {f}')
        f.unlink()


def get_plaats(plaats):
    url = f'https://geodata.nationaalgeoregister.nl/locatieserver/v3/free?fq=type:woonplaats&fl=*&q={plaats}&rows=10&outputFormat=json'
    
    req = requests.get(url)
    req.raise_for_status()
    
    data = req.json()
    
    if data['response']['numFound'] == 0:
        return None
    
    df_geo = gpd.GeoDataFrame(data['response']['docs'])
    
    return df_geo

def checkout_now_inzichtelijk(tmpdir, branch='master'):
    repo_clone_url = "https://github.com/ansien/now-inzichtelijk.git"
    local_repo = tmpdir.name
    repo = git.Repo.clone_from(repo_clone_url, local_repo)
    repo.git.checkout(branch)
    repo.git.pull()

    return repo.head.object.hexsha


def load_now_dfs(tmpdir):
    now_dfs = {}
    for nowfile in tqdm((Path(tmpdir.name) / 'public/file').glob('*.csv')):
        df_now = pd.read_csv(nowfile, sep=';')

        numerical_cols = [
            'VERSTREKT VOORSCHOT',
            'VASTGESTELDE SUBSIDIE',
        ]

        for col in numerical_cols:
            if col in df_now.columns:
                df_now[col] = df_now[col].str.replace(',', '').str.replace('-', '').apply(lambda x: int(x) if len(x) > 0 else float('NaN')).astype('Int64')

        now_dfs[nowfile] = df_now

    return now_dfs


def get_vestigingsplaatsen(now_dfs):
    vestigingsplaatsen = set()

    for df_now in now_dfs.values():
        vestigingsplaatsen |= set(df_now['VESTIGINGSPLAATS'].dropna().unique())

    return list(vestigingsplaatsen)


def resolve_vestigingsplaatsen(vestigingsplaatsen):
    mappings = {}
    unmapped = []
    multiresults = {}

    for vestigingsplaats in tqdm(vestigingsplaatsen, unit=' vestigingsplaatsen'):
        if vestigingsplaats.endswith('gn'):  # groningen
            vestigingsplaats = vestigingsplaats[:-3] + ' gr'
        elif vestigingsplaats.endswith('gld'):  # gelderland
            vestigingsplaats = vestigingsplaats[:-4] + ' gd'

        if vestigingsplaats in mappings:
            continue

        df_plaatsen = get_plaats(vestigingsplaats)

        if df_plaatsen is None:
            print(f'WARN: unmappable: {vestigingsplaats}')
            unmapped.append(vestigingsplaats)
            continue

        df_plaatsen['dist'] = df_plaatsen['woonplaatsnaam'].apply(lambda x: levenshtein.distance(x.lower(), vestigingsplaats))
        df_plaatsen = df_plaatsen[df_plaatsen['dist'] == df_plaatsen['dist'].min()]

        if df_plaatsen.shape[0] != 1:
            df_plaatsen['dist'] = df_plaatsen[['dist', 'provincieafkorting']].apply(lambda x: x['dist'] - 3 if vestigingsplaats.upper().endswith(f" {x['provincieafkorting']}") else x['dist'], axis=1)

        df_plaatsen = df_plaatsen[df_plaatsen['dist'] == df_plaatsen['dist'].min()]

        multiresults[vestigingsplaats] = df_plaatsen.copy()

        # pick the first one as it is the most likely match
        df_plaatsen = df_plaatsen.iloc[0].to_frame().T

        mappings[vestigingsplaats] = df_plaatsen

    return mappings, unmapped, multiresults


def augment_df_now(df_now, mappings, unmapped):
    df_aug = df_now.copy()

    vplats = df_aug['VESTIGINGSPLAATS'].fillna('').str.upper()
    
    df_aug['waarsch_buitenland'] = vplats.apply(lambda x: 1 if x in unmapped else 0)
    df_aug['provincie'] = vplats.apply(lambda x: None if x not in mappings or mappings[x] is None else mappings[x.upper()].iloc[0]['provincienaam'])
    df_aug['provincie_code'] = vplats.apply(lambda x: None if x not in mappings or mappings[x] is None else mappings[x.upper()].iloc[0]['provinciecode'])

    df_aug['gemeente'] = vplats.apply(lambda x: None if x not in mappings or mappings[x] is None else mappings[x.upper()].iloc[0]['gemeentenaam'])
    df_aug['gemeente_code'] = vplats.apply(lambda x: None if x not in mappings or mappings[x] is None else mappings[x.upper()].iloc[0]['gemeentecode'])

    return df_aug


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def main():
    tmpdir = tempfile.TemporaryDirectory(prefix='ni-aug-')

    print('Checkout now inzichtelijk...')
    latest_commit = checkout_now_inzichtelijk(tmpdir)
    print(f'Latest commit - {latest_commit} -')


    with open('data/latest_commit_now_inzichtelijk', 'r') as fh:
        comp_latest_registered_commit = fh.read().strip()

    print(f'Latest registered commit - {comp_latest_registered_commit} -')
    
    if comp_latest_registered_commit == latest_commit:
        print('No change, exiting.')
        return

    cleanup()

    now_dfs = load_now_dfs(tmpdir)
    vestigingsplaatsen = get_vestigingsplaatsen(now_dfs)
    mappings, unmapped, multiresults = resolve_vestigingsplaatsen(vestigingsplaatsen)

    mappings_serializable = {k: v.to_dict(orient='index') for k, v in mappings.items()}
    with open(outdir / 'mappings.json', 'w') as fh:
        json.dump(mappings_serializable, fh, cls=NpEncoder)

    with open(outdir / 'unmapped.json', 'w') as fh:
        json.dump(unmapped, fh, cls=NpEncoder)

    for nowfile, df_now in tqdm(now_dfs.items(), unit=' augmented now files'):
        df_aug = augment_df_now(df_now, mappings, unmapped)
        df_aug.to_csv(outdir / f'augmented_{nowfile.name}', index=False)

    with open('data/latest_commit_now_inzichtelijk', 'w') as fh:
        fh.write(latest_commit)

if __name__ == '__main__':
    main()

