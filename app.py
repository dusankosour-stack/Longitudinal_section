import pandas as pd                               # práce s tabulkovými daty
import numpy as np                                # numerické výpočty a pole
import matplotlib.pyplot as plt                   # vykreslování grafů
from scipy.interpolate import griddata, interp1d  # interpolační funkce
from scipy.ndimage import gaussian_filter         # Gaussovo vyhlazení
from pykrige.ok import OrdinaryKriging            # Kriging interpolace
import streamlit as st                            # webové rozhraní
import math                                       # matematické funkce (floor, log10...)
import io                                         # pro uložení grafu do paměti místo souboru na disku
import time
from PIL import Image                               

#Favicon
icon = Image.open("icon.png")
st.set_page_config(page_title="Podélné profily", page_icon=icon)


# --- Načtení statických souborů ---
# @st.cache_data říká Streamlitu: výsledek této funkce si zapamatuj
# a při příští interakci uživatele ji nevolej znovu - vrať uložený výsledek
# Bez toho by se soubory načítaly znovu při každém kliknutí
@st.cache_data
def load_static_files():
    # read_excel načte Excel soubor do DataFrame
    attr = pd.read_excel('kody_nadrze_PMO.xlsx')
    attr['Kód'] = attr['Kód'].astype(str).str.strip()  # zajistíme string typ
    params = pd.read_excel('parametry.xlsx')
    return attr, params

# Zavolá funkci a rozbalí výsledek do dvou proměnných
attr_table, params_table = load_static_files()

# Filtruje řádky kde Zobrazovat == True a vezme jen sloupec Parametr
# .tolist() převede pandas Series na běžný Python seznam
parametry = params_table[params_table['Zobrazovat'] == True]['Parametr'].tolist()

#--------------------------------------------------------------------------------
def sluc_duplicity(df):
    """
    Sloučí komplementární duplicitní řádky.
    
    Duplicitní řádky vznikají když laboratoř zapíše část parametrů
    do samostatného řádku bez času a hloubky. Tento řádek se sloučí
    s odpovídajícím regulérním řádkem přes Název místa + Datum odběru.
    
    Pravidla:
    - Řádky s časem se NIKDY nespojují mezi sebou
    - Řádek bez času se spojí pouze pokud existuje právě jeden
      odpovídající řádek s časem pro stejné místo a datum
    - Pokud odpovídajících řádků je více nebo žádný, řádek se zahodí
    """
    
    # Rozdělíme DataFrame na řádky s časem a bez času
    df_s_casem = df[df['Čas odběru'].notna()].copy()
    df_bez_casu = df[df['Čas odběru'].isna()].copy()
    
    # Pokud nejsou žádné řádky bez času, není co slučovat
    if df_bez_casu.empty:
        return df_s_casem
    
    # Projdeme každý řádek bez času
    for idx, radek in df_bez_casu.iterrows():
        
        # Najdeme odpovídající řádky přes Název místa + Datum odběru
        maska = (
            (df_s_casem['Název místa'] == radek['Název místa']) &
            (df_s_casem['Datum odběru'] == radek['Datum odběru'])
        )
        nalezene = df_s_casem[maska]
        
        if nalezene.empty:
            # Žádný odpovídající řádek - zahodíme
            continue
        
        if len(nalezene) > 1:
            # Více odpovídajících řádků - nevíme ke kterému patří, zahodíme
            continue
        
        # Přesně jeden odpovídající řádek - sloučíme parametry
        # combine_first doplní chybějící hodnoty z radek do nalezeneho řádku
        df_s_casem.loc[nalezene.index[0]] = (
            nalezene.iloc[0].combine_first(radek)
        )
    
    return df_s_casem

#---------------------------------------------------------------------------
@st.cache_data
def load_data(uploaded_file):
    """
    Načte a vyčistí hlavní datový soubor.
    
    Postup:
    1. Načte Excel, oddělí názvy sloupců a jednotky
    2. Odstraní směsné vzorky
    3. Sloučí komplementární duplicitní řádky (Brno hráz)
    4. Odstraní řádky bez hloubky
    5. Převede hodnoty na čísla
    6. Joinuje s atributovou tabulkou
    """
    
    # Načteme druhý řádek pro jednotky
    jednotky_radek = pd.read_excel(uploaded_file, header=None, skiprows=1, nrows=1)

    # Načteme data - první řádek jako hlavička, druhý řádek přeskočíme
    # pandas automaticky ošetří duplicitní názvy sloupců (.1, .2...)
    df = pd.read_excel(uploaded_file, header=0, skiprows=[1])

    # Sestavíme slovník {parametr: jednotka}
    jednotky_dict = {
        df.columns[i]: str(jednotky_radek.iloc[0, i])
        for i in range(len(df.columns))
        if str(jednotky_radek.iloc[0, i]) not in ['nan', 'None']
    }

          
    # Přejmenujeme oba sloupce Místo odběru na srozumitelnější názvy
    df = df.rename(columns={
        'Místo odběru': 'Kód',           # první sloupec - kód místa (brn000)
        'Místo odběru.1': 'Název místa'  # druhý sloupec - dlouhý název
    })

    # Převedeme Kód na string - pandas ho mohl načíst jako float kvůli prázdným buňkám
    df['Kód'] = df['Kód'].astype(str).str.strip()

    # Odstraníme řádky kde Kód je 'nan' (prázdné buňky)
    df = df[df['Kód'] != 'nan']
    
    # Odstraníme směsné vzorky podle slova 'směs' v dlouhém názvu
    # ~ je negace, na=False zachová NaN jako False
    df = df[~df['Název místa'].str.contains('směs', case=False, na=False)]
    
    # Sloučíme komplementární duplicitní řádky
    # Musí proběhnout PŘED odstraněním řádků bez hloubky
    df = sluc_duplicity(df)
    
    # Odstraníme řádky bez hloubky - po sloučení duplicit
    df = df[df['Hloubka'].notna()]
    
    # Převod hodnot - ošetří <hodnota, čísla i nečitelné záznamy
    def convert_value(val):
        if isinstance(val, str):        # zpracujeme jen textové hodnoty
            val = val.strip()           # odstraníme bílé znaky
            if val.startswith('<'):     # hodnota pod mezí detekce
                try:
                    return float(val[1:].replace(',', '.')) / 2
                except:
                    return np.nan
            try:
                # převod na číslo - funguje pro kladná i záporná čísla
                return float(val.replace(',', '.'))
            except:
                # cokoliv nečitelné → NaN
                return np.nan
        return val                      # čísla a NaN vrátíme beze změny
    
    # Převádíme jen sloupce které mají obsahovat čísla
    # parametry pochází z whitelistu v parametry.xlsx
    sloupce_k_prevodu = parametry + ['Hloubka', 'Dno', 'Průhlednost']

    # Převedeme jen ty které skutečně existují v datech
    # pro případ že by některý parametr z whitelistu chyběl v datech
    sloupce_k_prevodu = [c for c in sloupce_k_prevodu if c in df.columns]

    df[sloupce_k_prevodu] = df[sloupce_k_prevodu].apply(
        lambda col: col.map(convert_value)
    )
    
    # Převedeme datum na datetime formát pro správné řazení
    df['Datum odběru'] = pd.to_datetime(df['Datum odběru'])

    # Joinujeme s atributovou tabulkou přes kód místa
    # how='left' zachová všechny řádky z df i bez shody v attr_table
    df = df.merge(attr_table, on='Kód', how='left')
    
    # Zkrátíme název profilu - vezmeme část za první čárkou
    # "Brno, střed" → "střed"
    df['Název profilu'] = df['Profil'].str.split(',', n=1).str[-1].str.strip()
    
    # Validace - upozornění na kódy bez záznamu v atributové tabulce
    chybejici = df[df['Nádrž'].isna()]['Kód'].unique()
    if len(chybejici) > 0:
        st.warning(f'Tyto kódy chybí v atributové tabulce: {chybejici}')
    
    return df, jednotky_dict

#----------------------------------------------------------------------------
def rozdel_do_baliku(df_den):
    """
    Rozdělí měření jednoho dne do balíků.
    Pracuje jen s řádky hloubky 0 (nebo nejmenší dostupné hloubky)
    aby různé časy různých hloubek nezpůsobily falešné balíky.
    """
    
    # Vezmeme jen nejmenší hloubku každého profilu - reprezentuje začátek měření
    df_hladina = df_den.loc[df_den.groupby('Kód')['Hloubka'].idxmin()]
    
    # Seřadíme podle času
    df_hladina = df_hladina.sort_values('Čas odběru')
    
    baliky_kody = []
    aktualni_balik = []
    videne_kody = set()
    
    for idx, radek in df_hladina.iterrows():
        kod = radek['Kód']
        
        if kod in videne_kody:
            baliky_kody.append(aktualni_balik)
            aktualni_balik = [kod]
            videne_kody = {kod}
        else:
            aktualni_balik.append(kod)
            videne_kody.add(kod)
    
    if aktualni_balik:
        baliky_kody.append(aktualni_balik)
    
    # Vrátíme seznam DataFrame - jeden za každý balík
    # Každý balík obsahuje všechny hloubky profilů které do něj patří
    return [df_den[df_den['Kód'].isin(kody)] for kody in baliky_kody]


#-----------------------------------------------------------------------------------
def priprav_data(df, nadrz, datum, balik=0):
    """
    Vyfiltruje data pro konkrétní nádrž, datum a balík měření.
    
    Parametry:
    - df: celý DataFrame s daty
    - nadrz: název nádrže (např. 'Brno')
    - datum: vybrané datum (datetime.date objekt)
    - balik: index balíku (0 = první, 1 = druhý...), default 0
    
    Vrátí:
    - profily: seznam tuplů (vzdálenost, data_profilu, název_profilu)
    - delka_nadrze: celková délka nádrže v metrech
    - pocet_baliku: počet balíků v daný den (pro UI)
    - cas_baliku: čas prvního záznamu vybraného balíku (pro UI)
    """



    
    # Filtrujeme podle nádrže
    df_nadrz = df[df['Nádrž'] == nadrz]
    
    # Filtrujeme podle data
    df_den = df_nadrz[df_nadrz['Datum odběru'].dt.date == datum]

    # Zjistíme jestli se nějaký profil měřil více než jednou v daný den
    pocty = df_den.groupby('Kód').size()
    vice_mereni = (pocty > df_den['Hloubka'].nunique()).any()

    if vice_mereni:
        baliky = rozdel_do_baliku(df_den)
    else:
        # Jen jeden balík - není potřeba rozdělovat
        baliky = [df_den]

    if df_den.empty:
        return [], 0, 0, None
    
    # Rozdělíme do balíků
    baliky = rozdel_do_baliku(df_den)
    
    # Vybereme požadovaný balík
    df_balik = baliky[balik]
    
    # Čas prvního záznamu balíku pro zobrazení v UI
    cas_baliku = df_balik['Čas odběru'].dropna().iloc[0] if df_balik['Čas odběru'].notna().any() else None
    
    # Sestavíme seznam profilů
    profily = []
    for kod, skupina in df_balik.groupby('Kód'):
        skupina = skupina.sort_values('Hloubka')
        vzdalenost = skupina['Vzdálenost od hráze'].iloc[0]
        nazev = skupina['Název profilu'].iloc[0]
        profily.append((vzdalenost, skupina, nazev))
    
    # Seřadíme podle vzdálenosti od hráze
    profily.sort(key=lambda x: x[0])
    
    # Celková délka nádrže
    delka_nadrze = df_den['Délka nádrže'].iloc[0]
    
    return profily, delka_nadrze, len(baliky), cas_baliku


#---------------------------------------------------------------------------
def nice_scale(data_min, data_max, n_levels):
    """
    Vypočítá hezká zaoblená čísla pro osy a legendu grafu.
    Například místo 0.0 - 10.3 vrátí 0 - 10 s krokem 1.
    
    Parametry:
    - data_min, data_max: rozsah dat
    - n_levels: požadovaný počet úrovní (přibližný)
    
    Vrátí:
    - new_min, new_max: zaoblený rozsah
    - step: krok mezi úrovněmi
    """
    
    if data_min > data_max:
        data_min, data_max = data_max, data_min  # prohodíme pokud jsou obráceně
    
    rng = data_max - data_min
    if rng == 0:
        rng = abs(data_max) if data_max != 0 else 1  # ošetříme nulový rozsah
    
    # Vypočítáme přibližný krok
    step0 = rng / (n_levels - 1)
    
    # Najdeme řád velikosti kroku (mocnina 10)
    exponent = np.floor(np.log10(step0))
    fraction = step0 / (10**exponent)
    
    # Zaokrouhlíme krok na hezké číslo
    if fraction < 1.5:      nice_fraction = 1
    elif fraction < 3:      nice_fraction = 2
    elif fraction < 7:      nice_fraction = 5
    else:                   nice_fraction = 10
    
    step = nice_fraction * (10**exponent)
    
    # Zaokrouhlíme min a max na násobky kroku
    new_min = np.floor(data_min / step) * step  # dolů na nejbližší násobek
    new_max = np.ceil(data_max / step) * step   # nahoru na nejbližší násobek
    
    return new_min, new_max, step


#----------------------------------------------------------------
def fill_missing_at_surface(df, measurement_params):
    """
    Doplní chybějící hodnoty na povrchu (hloubka 0).
    
    Pokud chybí hodnota na nejmenší hloubce profilu, doplní ji
    první dostupnou hodnotou z větší hloubky. Důvod: sonda ještě
    nestihla změřit než se začalo zaznamenávat, ale hodnota na
    povrchu je fyzikálně stejná jako první dostupná.
    
    Parametry:
    - df: DataFrame s daty jednoho balíku
    - measurement_params: seznam názvů měřených parametrů
    """
    
    def fill_group(group):
        # Seřadíme podle hloubky - nejmenší hloubka první
        group = group.sort_values('Hloubka')
        for col in measurement_params:
            if pd.isna(group.iloc[0][col]):  # chybí hodnota na povrchu
                non_nan = group[col].dropna()
                if not non_nan.empty:
                    # Doplníme první dostupnou hodnotu z větší hloubky
                    group.iloc[0, group.columns.get_loc(col)] = non_nan.iloc[0]
        return group
    
    # Aplikujeme fill_group na každou kombinaci datum + místo
    return df.groupby(
        ['Datum odběru', 'Kód'],
        group_keys=False
    ).apply(fill_group)


#------------------------------------------------------------------------
def interpolate_profile(depths, values, max_depth):
    """
    Zhustí vertikální profil na 200 bodů pro plynulé vykreslení.
    
    Měření jsou po 1 metru, ale pro hladký graf potřebujeme
    mnohem více bodů. Funkce interpoluje mezi naměřenými body
    a extrapoluje až ke dnu nádrže.
    
    Parametry:
    - depths: pole naměřených hloubek (např. [0, 1, 2, 3...])
    - values: pole naměřených hodnot (např. teploty)
    - max_depth: maximální hloubka nádrže (= dno)
    
    Vrátí:
    - pole 200 hodnot rovnoměrně rozložených od 0 do max_depth
    """
    
    depths = np.asarray(depths, dtype=float)  # převedeme na numpy pole
    values = np.asarray(values, dtype=float)
    
    # Odstraníme NaN hodnoty - interpolace s nimi neumí pracovat
    maska = ~np.isnan(values)
    depths = depths[maska]
    values = values[maska]
    
    if len(depths) == 0:
        # Žádná platná data - vrátíme pole NaN
        return np.full(200, np.nan)
    
    # 200 rovnoměrně rozložených hloubek od 0 do dna
    depths_fine = np.linspace(0, max_depth, 200)
    
    # np.interp interpoluje lineárně mezi body
    # mimo rozsah dat extrapoluje konstantně (poslední dostupnou hodnotou)
    values_interp = np.interp(depths_fine, depths, values)
    
    return values_interp


#-------------------------------------------------------------------------
def plot_water_reservoir_temperature(ax, profily, delka_nadrze, param, metoda='linear'):
    """
    Vykreslí 2D barevný řez nádrže pro zvolený parametr.
    
    Postup:
    1. Připraví vstupní body pro interpolaci podle zvolené metody
    2. Interpoluje hodnoty na pravidelnou 2D mřížku
    3. Vyhladi mřížku Gaussovým filtrem
    4. Aplikuje masku dna
    5. Vykreslí barevnou mapu a konturové čáry
    
    Parametry:
    - ax: matplotlib osa pro vykreslení
    - profily: seznam tuplů (vzdálenost, data_profilu, název_profilu)
    - delka_nadrze: celková délka nádrže v metrech
    - param: název parametru k zobrazení (např. 'Teplota vody')
    - metoda: interpolační metoda ('linear', 'kriging')
    
    Vrátí:
    - im: objekt barevné mapy (pro colorbar)
    - boundaries: pole hranic pro colorbar
    - cbar_fmt: formátovací řetězec pro popisky colorbaru
    """

    if not profily:
        raise ValueError("Alespoň jeden profil je vyžadován")

    # Maximální hloubka = největší hodnota dna přes všechny profily
    max_depth = max(skupina['Dno'].max() for _, skupina, _ in profily)

    # Definice mřížky pro interpolaci. Pokud byl změřen jen 1 profil, vybarví sejen část nádrže
    if len(profily) == 1:
        pos = profily[0][0]
        sirka = delka_nadrze / 5
        x_min = max(0, pos - sirka)
        x_max = min(delka_nadrze, pos + sirka)
        x_grid = np.linspace(x_min, x_max, 320)
    else:
        x_grid = np.linspace(0, delka_nadrze, 320)

    z_grid = np.linspace(0, max_depth, 200)
    x_mesh, z_mesh = np.meshgrid(x_grid, z_grid)

    # --- INTERPOLACE ---
    if len(profily) == 1:
        # Jeden profil - vykreslíme jako svislý barevný pruh bez interpolace
        values_interp = interpolate_profile(
            profily[0][1]['Hloubka'].values,
            profily[0][1][param].values,
            max_depth
        )
        # Rozšíříme na 2D zopakováním profilu přes celou šířku
        temp_grid = np.tile(values_interp.reshape(-1, 1), (1, 320))

    # --- CESTA A: LINEAR ---
    elif metoda == 'linear':
        points, values = [], []
        profily_s_daty = []  # seznam profilů které mají data
        
        for pos, skupina, nazev in profily:
            d_arr = skupina['Hloubka'].values
            v_arr = skupina[param].values
            maska = ~np.isnan(v_arr)
            if not np.any(maska):
                continue
            
            depths_fine = np.linspace(0, max_depth, 200)
            values_interp = interpolate_profile(d_arr[maska], v_arr[maska], max_depth)
            points.extend([(pos, z) for z in depths_fine])
            values.extend(values_interp)
            profily_s_daty.append((pos, skupina, nazev))
        
        # Pokud má data jen jeden profil, zobrazíme jako pruh
        if len(profily_s_daty) == 1:
            pos_s_daty = profily_s_daty[0][0]
            sirka = delka_nadrze / 5
            x_min = max(0, pos_s_daty - sirka)
            x_max = min(delka_nadrze, pos_s_daty + sirka)
            x_grid = np.linspace(x_min, x_max, 320)
            x_mesh, z_mesh = np.meshgrid(x_grid, z_grid)  # přepočítáme mřížku
            
            d_arr = profily_s_daty[0][1]['Hloubka'].values
            v_arr = profily_s_daty[0][1][param].values
            maska = ~np.isnan(v_arr)
            values_interp = interpolate_profile(d_arr[maska], v_arr[maska], max_depth)
            temp_grid = np.tile(values_interp.reshape(-1, 1), (1, 320))
        else:
            # Přidáme krajní profily
            if len(profily) > 1:
                for edge_x in [0, delka_nadrze]:
                    ref = profily[0] if edge_x == 0 else profily[-1]
                    d_arr = ref[1]['Hloubka'].values
                    v_arr = ref[1][param].values
                    maska = ~np.isnan(v_arr)
                    if np.any(maska):
                        depths_fine = np.linspace(0, max_depth, 200)
                        values_interp = interpolate_profile(d_arr[maska], v_arr[maska], max_depth)
                        points.extend([(edge_x, z) for z in depths_fine])
                        values.extend(values_interp)
            
            points = np.array(points)
            values = np.array(values)
            temp_grid = griddata(points, values, (x_mesh, z_mesh), method='linear')
    
    # --- CESTA B: KRIGING ---
    elif metoda == 'kriging':
        points_k, values_k = [], []
        
        # Pro Kriging bereme přímo naměřené body - žádné zhušťování
        for pos, skupina, _ in profily:
            d_arr = skupina['Hloubka'].values
            v_arr = skupina[param].values
            dno_val = skupina['Dno'].max()
            
            maska = ~np.isnan(v_arr)
            if np.any(maska):
                # Přidáme všechny naměřené body
                for d, v in zip(d_arr[maska], v_arr[maska]):
                    points_k.append((pos, d))
                    values_k.append(v)
                
                # Přidáme bod na dně pokud poslední měření není na dně
                # zajistí že interpolace sahá až ke dnu
                if d_arr[maska][-1] < dno_val:
                    points_k.append((pos, dno_val))
                    values_k.append(v_arr[maska][-1])
        
        # Přidáme krajní body pro interpolaci k okrajům nádrže
        if len(profily) > 1:
            for edge_x in [0, delka_nadrze]:
                ref = profily[0] if edge_x == 0 else profily[-1]
                d_arr = ref[1]['Hloubka'].values
                v_arr = ref[1][param].values
                dno_val = ref[1]['Dno'].max()
                maska = ~np.isnan(v_arr)
                if np.any(maska):
                    # Zkopírujeme všechny naměřené body krajního profilu
                    for d, v in zip(d_arr[maska], v_arr[maska]):
                        points_k.append((edge_x, d))
                        values_k.append(v)
                    # Přidáme bod na dně
                    if d_arr[maska][-1] < dno_val:
                        points_k.append((edge_x, dno_val))
                        values_k.append(v_arr[maska][-1])
        
        points_k = np.array(points_k)
        values_k = np.array(values_k)
        
        # Ošetření duplicit - průměrujeme body se stejnými souřadnicemi
        temp_df = pd.DataFrame({
            'x': points_k[:, 0],
            'z': points_k[:, 1],
            'v': values_k
        })
        temp_df = temp_df.groupby(['x', 'z']).mean().reset_index()
        
        try:
            # Škálování souřadnic - kompenzuje velký rozdíl mezi
            # horizontálním (stovky metrů) a vertikálním (jednotky metrů)
            # rozměrem nádrže. Bez škálování by Kriging ignoroval
            # vertikální strukturu dat.
            scaling_factor = delka_nadrze / max(max_depth, 1.0)
            scaling_factor = np.clip(scaling_factor, 10, 1000)
            
            ok = OrdinaryKriging(
                temp_df['x'].values,
                temp_df['z'].values * scaling_factor,  # škálujeme vertikálu
                temp_df['v'].values,
                variogram_model='spherical',
                variogram_parameters={
                    'nugget': 0.1,
                    'range': delka_nadrze / 2,
                    'sill': np.var(values_k)
                },
                verbose=False,
                enable_plotting=False
            )
            
            # execute vypočítá hodnoty na celé mřížce
            temp_grid, ss = ok.execute(
                'grid',
                x_grid,
                z_grid * scaling_factor,  # škálujeme i mřížku
                backend='vectorized'
            )
            temp_grid = np.clip(temp_grid, np.nanmin(values_k), np.nanmax(values_k))
            
        except Exception as e:
            # Záchranná větev - při selhání Krigingu přepneme na Linear
            st.warning(f"Kriging selhal, přepínám na Linear: {e}")
            temp_grid = griddata(
                points_k, values_k, (x_mesh, z_mesh), method='linear'
            )
    
    # Gaussovo vyhlazení - odstraní drobné artefakty interpolace
    # sigma=1 je mírné vyhlazení, větší hodnota = více rozmazání
    temp_grid = gaussian_filter(temp_grid, sigma=1)
    
    # --- MASKA DNA ---
    # Interpolujeme průběh dna mezi profily
    bottom_x = [0] + [p[0] for p in profily] + [delka_nadrze]
    bottom_z = (
        [profily[0][1]['Dno'].values[-1]] +
        [p[1]['Dno'].values[-1] for p in profily] +
        [0]
    )
    
    # interp1d vytvoří interpolační funkci průběhu dna
    bottom_interp = interp1d(bottom_x, bottom_z, kind='linear')
    
    # Vytvoříme masku - True kde je voda, False kde je dno nebo vzduch
    maska = np.zeros_like(temp_grid, dtype=bool)
    for i, x in enumerate(x_grid):
        maska[:, i] = z_grid <= bottom_interp(x)  # True pro hloubky nad dnem
    
    # Aplikujeme masku na mřížku - hodnoty pod dnem se nezobrazí
    temp_grid_masked = np.ma.array(temp_grid, mask=~maska)

    # --- VYKRESLENÍ ---
    # imshow zobrazí 2D pole jako barevný obraz
    # aspect='auto' přizpůsobí poměr stran grafu
    # extent definuje rozsah os
    im = ax.imshow(
        temp_grid_masked,
        aspect='auto',
        extent=[x_grid[0], x_grid[-1], max_depth, 0]
    )
    
    # Výpočet hranic pro colorbar a konturové čáry
    n_levels = 10
    d_min = np.nanmin(temp_grid_masked)
    d_max = np.nanmax(temp_grid_masked)

    # Pokud jsou všechny hodnoty stejné nebo velmi podobné
    if d_max - d_min < 1e-10:
        d_min = d_min - 0.1
        d_max = d_max + 0.1

    new_min, new_max, step = nice_scale(d_min, d_max, n_levels)
    boundaries = np.arange(new_min, new_max + step, step)

    # Pojistka - boundaries musí mít aspoň 2 hodnoty a být monotonní
    if len(boundaries) < 2:
        boundaries = np.linspace(d_min, d_max, n_levels)
    
    # Formátování popisků - počet desetinných míst podle kroku
    decimals = max(0, -int(math.floor(math.log10(step)))) if step > 0 else 0
    clabel_fmt = f"%1.{decimals}f"
    cbar_fmt = f"{{:.{decimals}f}}"

    # Konturové čáry - u palety binary barva závisí na jasu pozadí
    cmap_obj = plt.colormaps[cmap]

    if 'binary' in cmap:
        c_colors = []
        for v in boundaries:
            norm_v = (v - new_min) / (new_max - new_min) if new_max != new_min else 0
            rgba = cmap_obj(norm_v)
            jas = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            c_colors.append('white' if jas < 0.4 else 'black')
    else:
        c_colors = ['black'] * len(boundaries)

    contours = ax.contour(
        x_mesh, z_mesh, temp_grid_masked,
        levels=boundaries,
        colors=c_colors,
        linewidths=0.8,
        alpha=0.8
    )
    ax.clabel(contours, inline=True, fmt=clabel_fmt, fontsize=10)
    
    # Body měření
    for pos, skupina, _ in profily:
        ax.plot(
            [pos] * len(skupina),
            skupina['Hloubka'].values,
            'k.',
            markersize=2
        )
    
    # Průběh dna
    ax.plot(
        np.linspace(0, delka_nadrze, 100),
        bottom_interp(np.linspace(0, delka_nadrze, 100)),
        'k--',
        linewidth=1
    )
    
    # Popis os
    ax.set_xlabel('Vzdálenost od hráze [m]')
    ax.set_ylabel('Hloubka [m]')
    ax.grid(True)
    
    # Horní osa s názvy profilů
    ax_top = ax.secondary_xaxis('top')
    ax_top.set_xticks([p[0] for p in profily])
    ax_top.set_xticklabels(
        [p[2] for p in profily],
        fontsize=10,
        rotation=45,  # pootočení pro lepší čitelnost
        ha='left'
    )
    
    return im, boundaries, cbar_fmt

#----------------------------------------------------------------------
# --- STREAMLIT UI ---

st.markdown("""
    <style>
        .block-container {
            padding-top: 2.5rem;
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 90%;
        }
        section[data-testid="stSidebar"] > div:first-child {
            padding-top: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- LEVÁ LIŠTA ---
with st.sidebar:
    st.markdown("## Podélný profil nádrže")
    
    uploaded_file = st.file_uploader("Nahrajte datový soubor", type="xlsx")
    if uploaded_file is None:
        st.info("Nahrajte datový soubor.")
        st.stop()
    
    df, jednotky_dict = load_data(uploaded_file)
    
    nadrze_v_datech = df['Nádrž'].dropna().unique().tolist()
    nadrze = [n for n in attr_table['Nádrž'].unique().tolist() if n in nadrze_v_datech]
    nadrz = st.selectbox("Nádrž", nadrze)
    df_nadrz = df[df['Nádrž'] == nadrz]
    
    metoda = st.radio("Interpolace", ["linear", "kriging"], horizontal=True)

    PARAMETRY_SPECTRAL = ['Rozpuštěný kyslík', '% kyslíku', 'pH terén', 'ORP']
    param_pro_cmap = st.session_state.get('param_vyber', parametry[0])
    default_cmap = 'Spectral' if param_pro_cmap in PARAMETRY_SPECTRAL else 'Spectral_r'
    cmap = st.radio(
        "Barevná paleta",
        ["Spectral_r", "Spectral", "binary_r", "binary"],
        index=["Spectral_r", "Spectral", "binary_r", "binary"].index(default_cmap),
        horizontal=True
    )

    with st.popover("ℹ️ O metodách interpolace"):
        st.markdown("""
    **Linear (griddata)**  
    Každý vertikální profil se nejdřív zhustí na 200 bodů lineární 
    interpolací a extrapoluje se ke dnu nádrže. Krajní profily se 
    nakopírují na hráz a ústí nádrže. Teprve pak se provede 
    2D horizontální interpolace na pravidelnou mřížku.

    **Kriging (Ordinary Kriging)**  
    Geostatistická metoda, která odhaduje prostorovou autokorelaci dat 
    pomocí variogramu a na jeho základě interpoluje. Lépe respektuje 
    vertikální vrstvení nádrže. Používá surová data po 1 metru bez 
    předchozího zhušťování.

    Souřadnice jsou před Krigingem škálovány poměrem délky a hloubky 
    nádrže, aby se kompenzoval velký rozdíl horizontálního a vertikálního 
    rozměru. Může generovat nepřirozená minima a maxima u parametrů s malou prostorovou 
    variabilitou (pH, konduktivita) nebo při malém počtu měřených profilů.
        """)

# --- HLAVNÍ PLOCHA ---
datumy = sorted(df_nadrz['Datum odběru'].dt.date.unique().tolist())

if not datumy:
    st.warning("Pro vybranou nádrž nejsou k dispozici žádná data.")
    st.stop()

datum = st.select_slider(
    "Datum",
    options=datumy,
    format_func=lambda d: str(d)
)

df_den = df_nadrz[df_nadrz['Datum odběru'].dt.date == datum]

parametry_dostupne = [
    p for p in parametry
    if p in df_den.columns and df_den[p].notna().any()
]

if not parametry_dostupne:
    st.warning("Pro vybraný den nejsou k dispozici žádné parametry.")
    st.stop()

# Inicializace vybraného parametru
if 'param_vyber' not in st.session_state or st.session_state.param_vyber not in parametry_dostupne:
    st.session_state.param_vyber = parametry_dostupne[0]

# Výběr balíku
profily, delka_nadrze, pocet_baliku, _ = priprav_data(df, nadrz, datum, balik=0)

if pocet_baliku > 1:
    baliky_moznosti = []
    for i in range(pocet_baliku):
        _, _, _, cas = priprav_data(df, nadrz, datum, balik=i)
        cas_str = pd.to_datetime(cas).strftime("%H:%M") if cas and not pd.isna(cas) else f"Balík {i+1}"
        baliky_moznosti.append(cas_str)
    balik_idx = st.selectbox(
        "Měření",
        options=range(pocet_baliku),
        format_func=lambda i: baliky_moznosti[i]
    )
    profily, delka_nadrze, _, _ = priprav_data(df, nadrz, datum, balik=balik_idx)
else:
    balik_idx = 0

# --- ANIMACE ---
if 'animace_bezi' not in st.session_state:
    st.session_state.animace_bezi = False
if 'animace_idx' not in st.session_state:
    st.session_state.animace_idx = 0

col1, col2 = st.columns(2)
with col1:
    if st.button("▶ Přehrát animaci"):
        st.session_state.animace_bezi = True
        st.session_state.animace_idx = 0
with col2:
    if st.button("⏹ Zastavit"):
        st.session_state.animace_bezi = False

if st.session_state.animace_bezi:
    param = st.session_state.param_vyber
    idx = st.session_state.animace_idx
    
    # Při prvním snímku spočítáme globální rozsah pro celou sezónu
    if idx == 0:
        vsechny_hodnoty = []
        for datum_tmp in datumy:
            df_tmp = df[(df['Nádrž'] == nadrz) &
                        (df['Datum odběru'].dt.date == datum_tmp)]
            if param in df_tmp.columns:
                vsechny_hodnoty.extend(df_tmp[param].dropna().tolist())
        if vsechny_hodnoty:
            g_min, g_max, g_step = nice_scale(min(vsechny_hodnoty), max(vsechny_hodnoty), 10)
            st.session_state.global_boundaries = np.arange(g_min, g_max + g_step, g_step)
        else:
            st.session_state.global_boundaries = None
    
    if idx < len(datumy):
        datum_anim = datumy[idx]
        profily_anim, delka_anim, _, _ = priprav_data(df, nadrz, datum_anim, balik=0)
        df_den_anim = df[(df['Nádrž'] == nadrz) &
                         (df['Datum odběru'].dt.date == datum_anim)]
        if profily_anim and df_den_anim[param].notna().sum() > 0:
            fig_anim, (ax_anim, ax_profil_anim) = plt.subplots(
                1, 2,
                figsize=(14, 5),
                gridspec_kw={'width_ratios': [2, 1]},
                sharey=True
)
            try:
                im, boundaries, cbar_fmt = plot_water_reservoir_temperature(
                    ax_anim, profily_anim, delka_anim, param=param, metoda=metoda
                )
                im.set_cmap(plt.colormaps[cmap])
                # Použijeme globální rozsah pokud je k dispozici
                if st.session_state.global_boundaries is not None:
                    boundaries = st.session_state.global_boundaries
                    # Přepočítáme cbar_fmt pro globální rozsah
                    g_step = boundaries[1] - boundaries[0]
                    decimals = max(0, -int(math.floor(math.log10(g_step)))) if g_step > 0 else 0
                    cbar_fmt = f"{{:.{decimals}f}}"
                
                im.set_clim(vmin=boundaries[0], vmax=boundaries[-1])
                jednotka = jednotky_dict.get(param, "")
                cmap_diskretni = plt.colormaps[cmap].resampled(len(boundaries) - 1)
                sm = plt.cm.ScalarMappable(
                    cmap=cmap_diskretni,
                    norm=plt.matplotlib.colors.BoundaryNorm(boundaries, cmap_diskretni.N)
                )
                sm.set_array([])
                cbar = fig_anim.colorbar(sm, ax=ax_anim, label=f"{param} [{jednotka}]")
                cbar.set_ticks(boundaries)
                cbar.set_ticklabels([cbar_fmt.format(b) for b in boundaries])
                ax_anim.set_title(
                    f"{param} — {nadrz} — {datum_anim.strftime('%d. %m. %Y')} ({idx + 1}/{len(datumy)})", pad=25
                )

                # Boční čárový graf
                cmap_profily = plt.colormaps['tab10']
                pozice = [p[0] for p in profily_anim]
                pos_min = min(pozice)
                pos_max = max(pozice) if max(pozice) != pos_min else pos_min + 1

                for pos, skupina, nazev in profily_anim:
                    d_arr = skupina['Hloubka'].values
                    v_arr = skupina[param].values
                    maska = ~np.isnan(v_arr)
                    if not np.any(maska):
                        continue
                    norm_pos = (pos - pos_min) / (pos_max - pos_min)
                    barva = cmap_profily(norm_pos)
                    ax_profil_anim.plot(v_arr[maska], d_arr[maska], color=barva, label=nazev, linewidth=2)
                    ax_profil_anim.plot(v_arr[maska], d_arr[maska], 'o', color=barva, markersize=4)

                ax_profil_anim.set_xlabel(f"{param} [{jednotka}]")
                ax_profil_anim.yaxis.set_tick_params(labelleft=False)
                ax_profil_anim.grid(True)
                ax_profil_anim.legend(fontsize=10, bbox_to_anchor=(1.01, 1), loc='upper left')
                for b in boundaries:
                    ax_profil_anim.axvline(x=b, color='gray', linewidth=0.3, alpha=0.5)

                fig_anim.tight_layout()
                st.pyplot(fig_anim, use_container_width=True)
            except Exception as e:
                st.warning(f"Přeskočen datum {datum_anim}: {e}")
            finally:
                plt.close(fig_anim)
        st.session_state.animace_idx += 1
        time.sleep(0.5)
        st.rerun()
    else:
        st.session_state.animace_bezi = False
        st.session_state.animace_idx = 0
        st.rerun()

else:
    # --- NORMÁLNÍ GRAF ---
    if not profily:
        st.warning("Pro vybraný den a nádrž nejsou k dispozici žádná data.")
        st.stop()

    param = st.session_state.param_vyber

    df_den_cisty = fill_missing_at_surface(df_den, parametry_dostupne)
    profily, delka_nadrze, _, _ = priprav_data(df, nadrz, datum, balik=balik_idx)

    fig, (ax, ax_profil) = plt.subplots(
        1, 2,
        figsize=(14, 5),
        gridspec_kw={'width_ratios': [2, 1]},
        sharey=True  # stejná osa y = hloubka
    )

    try:
        im, boundaries, cbar_fmt = plot_water_reservoir_temperature(
            ax, profily, delka_nadrze, param=param, metoda=metoda
        )
        im.set_cmap(plt.colormaps[cmap])
        im.set_clim(vmin=boundaries[0], vmax=boundaries[-1])
        jednotka = jednotky_dict.get(param, "")
        cmap_diskretni = plt.colormaps[cmap].resampled(len(boundaries) - 1)
        sm = plt.cm.ScalarMappable(
            cmap=cmap_diskretni,
            norm=plt.matplotlib.colors.BoundaryNorm(boundaries, cmap_diskretni.N)
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, label=f"{param} [{jednotka}]")
        cbar.set_ticks(boundaries)
        cbar.set_ticklabels([cbar_fmt.format(b) for b in boundaries])
        ax.set_title(
            f"{param} — {nadrz} — {datum.strftime('%d. %m. %Y')}", pad=25
        )

        # --- BOČNÍ ČÁROVÝ GRAF ---
        # Barvy profilů podle vzdálenosti od hráze
        cmap_profily = plt.colormaps['tab10']
        pozice = [p[0] for p in profily]
        pos_min = min(pozice)
        pos_max = max(pozice) if max(pozice) != pos_min else pos_min + 1

        for pos, skupina, nazev in profily:
            d_arr = skupina['Hloubka'].values
            v_arr = skupina[param].values
            maska = ~np.isnan(v_arr)
            if not np.any(maska):
                continue
            # Normalizujeme pozici na 0-1 pro colormapu
            norm_pos = (pos - pos_min) / (pos_max - pos_min)
            barva = cmap_profily(norm_pos)
            ax_profil.plot(v_arr[maska], d_arr[maska], color=barva, label=nazev, linewidth=2)
            ax_profil.plot(v_arr[maska], d_arr[maska], 'o', color=barva, markersize=4)

        ax_profil.set_xlabel(f"{param} [{jednotka}]")
        ax_profil.set_ylabel("")  # sdílí osu y s hlavním grafem
        ax_profil.yaxis.set_tick_params(labelleft=False)
        ax_profil.grid(True)
        ax_profil.legend(fontsize=10, bbox_to_anchor=(1.01, 1), loc='upper left')

        # Přidáme vertikální čáry pro boundaries aby odpovídaly contour plotu
        for b in boundaries:
            ax_profil.axvline(x=b, color='gray', linewidth=0.3, alpha=0.5)

        fig.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Chyba při vykreslování grafu: {e}")
    finally:
        plt.close(fig)

    # Výběr parametru pod grafem
    novy_param = st.radio(
        "Parametr", parametry_dostupne, horizontal=True,
        index=parametry_dostupne.index(st.session_state.param_vyber)
    )
    if novy_param != st.session_state.param_vyber:
        st.session_state.param_vyber = novy_param
        st.rerun()

    # --- STAŽENÍ GRAFU ---
    buf = io.BytesIO()
    fig_save, ax_save = plt.subplots(figsize=(14, 6))
    try:
        im, boundaries, cbar_fmt = plot_water_reservoir_temperature(
            ax_save, profily, delka_nadrze, param=param, metoda=metoda
        )
        im.set_cmap(plt.colormaps[cmap])
        im.set_clim(vmin=boundaries[0], vmax=boundaries[-1])
        jednotka = jednotky_dict.get(param, "")
        cmap_diskretni = plt.colormaps[cmap].resampled(len(boundaries) - 1)
        sm = plt.cm.ScalarMappable(
            cmap=cmap_diskretni,
            norm=plt.matplotlib.colors.BoundaryNorm(boundaries, cmap_diskretni.N)
        )
        sm.set_array([])
        cbar = fig_save.colorbar(sm, ax=ax_save, label=f"{param} [{jednotka}]")
        cbar.set_ticks(boundaries)
        cbar.set_ticklabels([cbar_fmt.format(b) for b in boundaries])
        ax_save.set_title(
            f"{param} — {nadrz} — {datum.strftime('%d. %m. %Y')}", pad=25
        )
        fig_save.savefig(buf, dpi=300, bbox_inches='tight')
        buf.seek(0)
        st.download_button(
            label="Stáhnout graf",
            data=buf,
            file_name=f"{nadrz}_{param}_{datum}.png",
            mime="image/png"
        )
    except Exception as e:
        st.error(f"Chyba při ukládání grafu: {e}")
    finally:
        plt.close(fig_save)

