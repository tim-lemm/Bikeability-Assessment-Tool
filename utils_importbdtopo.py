#import necessary libraries
from config import *
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import numpy as np
from pathlib import Path
import rasterio
from rasterio.merge import merge
from shapely.geometry import LineString


#Functions

## general function

def remplacer_nan_colonne(gdf, colonne, valeur_remplacement):
    """
    Remplace les valeurs NaN d'une colonne d'un GeoDataFrame par une valeur spécifiée.

    Paramètres :
        gdf (gpd.GeoDataFrame) : Le GeoDataFrame à traiter.
        colonne (str) : Le nom de la colonne à traiter.
        valeur_remplacement : La valeur à utiliser pour remplacer les NaN.

    Retour :
        gpd.GeoDataFrame : Un nouveau GeoDataFrame avec les NaN remplacés.
    """
    gdf_copy = gdf.copy()
    gdf_copy[colonne] = gdf_copy[colonne].fillna(valeur_remplacement)
    return gdf_copy

def flatten_cell(cell):
    """
    Flattens a cell by returning a single representative value if possible.

    If the input `cell` is a non-empty list where all elements are equal, returns the common element.
    If the first element of the list is 'nan' or None, returns that element.
    Otherwise, returns the input `cell` unchanged.

    Args:
        cell (Any): The input value, which may be a list or any other type.

    Returns:
        Any: The flattened value if possible, otherwise the original input.
    """
    if isinstance(cell, list) and len(cell) > 0:
        if all(x == cell[0] for x in cell):
            return cell[0]
        elif cell[0] == 'nan' or cell[0] is None:
            return cell[0]
    return cell

def flatten_identical_lists(gdf):
    """
    Pour chaque colonne du GeoDataFrame, si la valeur est une liste dont tous les éléments sont identiques,
    remplace la liste par la valeur unique.

    Paramètres :
        gdf (gpd.GeoDataFrame) : Le GeoDataFrame à traiter.

    Retour :
        gpd.GeoDataFrame : Un nouveau GeoDataFrame avec les listes aplaties si possible.
    """
    gdf_flat = gdf.copy()
    for col in gdf_flat.columns:
        if col != 'geometry':
            gdf_flat[col] = gdf_flat[col].apply(flatten_cell)
    return gdf_flat

def fusionner_par_attribut(gdf, attribut):
    """
    Fusionne les lignes et les géométries d'un GeoDataFrame selon un attribut donné.
    Les autres colonnes sont agrégées sous forme de listes.

    Parameters:
        gdf (gpd.GeoDataFrame): Le GeoDataFrame à fusionner.
        attribut (str): Le nom de la colonne sur laquelle faire la fusion.

    Returns:
        gpd.GeoDataFrame: Un nouveau GeoDataFrame fusionné.
    """
    grouped = gdf.dissolve(by=attribut, as_index=False)
    # Pour les autres colonnes, on peut agréger en listes si besoin :
    for col in gdf.columns:
        if col not in [attribut, 'geometry']:
            grouped[col] = gdf.groupby(attribut)[col].agg(list).values
    grouped = flatten_identical_lists(grouped)
    return grouped

def fix_gdf(gdf, 
            list_supp = ['diff_altitude_rgalti','altitude_fin_rgalti','altitude_debut_rgalti','delestage','restriction_de_hauteur','prive','periode_de_fermeture']
            ):
    """
    Supprime les colonnes list_supp du GeoDataFrame et additionne les listes dans la colonne 'length'.

    Paramètres :
        gdf (gpd.GeoDataFrame) : Le GeoDataFrame à traiter.
        list_supp (list) : Liste des noms de colonnes à supprimer.

    Retour :
        gpd.GeoDataFrame : Le GeoDataFrame modifié.
    """
    gdf = gdf.copy()
    # Additionner les listes dans la colonne 'length'
    gdf['length'] = gdf['length'].apply(lambda x: np.sum(x) if isinstance(x, list) else x)
    # Supprimer les colonnes
    gdf = gdf.drop(columns=list_supp, errors='ignore')
    return gdf

## zones exclues
def set_zones_exclues(row):
    """
    Set the 'exclue' column based on the 'nature' and 'nature_detaillee' columns.
    """
    nature = row.get('nature', '')
    nature_detaillee = row.get('nature_detaillee', '')
    if (nature == "Espace public") and (nature_detaillee in list_zones_exclues_nature_detaillee):
        return True
    elif nature in list_zones_exclues_nature:
        return True
    return False

## sentier
def set_amenagement_sentier_if_nature_sentier(gdf):
    mask = gdf['nature'].str.lower() == 'sentier'
    gdf.loc[mask, 'amenagement_cyclable_gauche'] = 'Sentier'
    gdf.loc[mask, 'amenagement_cyclable_droit'] = 'Sentier'
    gdf.loc[mask, 'nature_de_la_restriction']= 'Sentier'

## amenagement from restriction
def get_restriction_value(row):
    val = dict_nature_de_la_restriction.get(row.get('nature_de_la_restriction'),0)
    return val

def get_restriction_value_gauche(row):
    val = 0
    if row.get('restriction_val') == 0:
        val = dict_amenagment_cyclable.get(row.get('amenagement_cyclable_gauche'), 0)
    return val

def get_restriction_value_droit(row):
    val = 0
    if row.get('restriction_val') == 0:
        val = dict_amenagment_cyclable.get(row.get('amenagement_cyclable_droit'), 0)
    return val

def get_presence_amenagment(row):
    val = True
    if row.get('restriction_val') == 0 and row.get('restriction_val_gauche') == 0 and row.get('restriction_val_droit') == 0 :
        val = False
    return val

def get_name_amenagement(row):
    return dict_categories_amenagement_cyclable.get(row.get('restriction_val'), "Pas d'aménagement cyclable") 

def get_name_amenagement_droit(row):
    val = dict_categories_amenagement_cyclable.get(row.get('restriction_val_droit'), "Pas d'aménagement cyclable")
    if val == "Pas d'aménagement cyclable" and row.get('nom_amenagement') != "Pas d'aménagement cyclable":
        val = row.get('nom_amenagement')
    return val

def get_name_amenagement_gauche(row):
    val = dict_categories_amenagement_cyclable.get(row.get('restriction_val_gauche'), "Pas d'aménagement cyclable")
    if val == "Pas d'aménagement cyclable" and row.get('nom_amenagement') != "Pas d'aménagement cyclable":
        val = row.get('nom_amenagement')
    return val 

def correct_pistes (row):
    if row['sens_amenagement_cyclable_droit'] == 'Double Sens':
        row['amenagement_cyclable_gauche'] = row['amenagement_cyclable_droit']
    if row['sens_amenagement_cyclable_gauche'] == 'Double Sens':
        row['amenagement_cyclable_droit'] = row['amenagement_cyclable_gauche']

def apply_amenagment(gdf):
    set_amenagement_sentier_if_nature_sentier(gdf)
    gdf['restriction_val'] = gdf.apply(get_restriction_value, axis=1)
    gdf['restriction_val_gauche'] = gdf.apply(get_restriction_value_gauche, axis=1)
    gdf['restriction_val_droit'] = gdf.apply(get_restriction_value_droit, axis=1)
    gdf['presence_amenagement'] = gdf.apply(get_presence_amenagment, axis =1)
    gdf['nom_amenagement'] = gdf.apply(get_name_amenagement, axis =1)
    gdf['nom_amenagement_droit'] = gdf.apply(get_name_amenagement_droit, axis =1)
    gdf['nom_amenagement_gauche'] = gdf.apply(get_name_amenagement_gauche, axis =1)
    gdf.apply(correct_pistes, axis=1)
    
#slopes functions

def import_and_merge_asc_for_city(city_polygon, 
                                  file_path_dalles = 'data/IGN/RGEALTI/dalles.shp',
                                  file_path_dalles_asc = 'data/IGN/RGEALTI/raster/'):
    """
    Importe et fusionne les fichiers .asc RGALTI correspondant à un polygone de ville.
    Retourne un objet rasterio (dataset fusionné) et le tableau numpy associé.
    """
    # Trouver les dalles intersectant la ville
    gdf_dalles = gpd.read_file(file_path_dalles)
    dalles = gdf_dalles[gdf_dalles.intersects(city_polygon)]
    noms_dalles = dalles['NOM_DALLE'].tolist()
    asc_files = [file_path_dalles_asc + nom + '.asc' for nom in noms_dalles]
    asc_files = [f for f in asc_files if Path(f).exists()]
    if not asc_files:
        print("Aucun fichier .asc trouvé pour la ville.")
        return None, None

    src_files_to_mosaic = [rasterio.open(f) for f in asc_files]
    mosaic, out_trans = merge(src_files_to_mosaic)
    # On retourne le tableau numpy et le profil raster
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "crs": src_files_to_mosaic[0].crs
    })
    for src in src_files_to_mosaic:
        src.close()
    return mosaic, out_meta

def xy_to_rowcol(x, y, out_meta):
        transform = out_meta['transform']
        col, row = ~transform * (x, y)
        return int(round(row)), int(round(col))

def safe_get(row, col, h, w, mosaic, out_meta):
        if 0 <= row < h and 0 <= col < w:
            val = mosaic[0, row, col]
            if val != out_meta.get('nodata', -99999):
                return val
        return None

def get_altitude_difference_from_rgalti(geom, mosaic, out_meta, opt_abs=True):
    """
    Calcule la différence d'altitude (z_fin - z_debut) pour un tronçon (LineString)
    à partir du MNT RGALTI (mosaic, out_meta).
    geom : shapely LineString (2D ou 3D)
    mosaic : numpy array du MNT (1, H, W)
    out_meta : dict rasterio (doit contenir 'transform')
    Retourne : (alt_debut, alt_fin, diff)
    """

    if not isinstance(geom, LineString):
        raise ValueError("La géométrie doit être une LineString.")

    # Récupérer les coordonnées du début et de la fin
    coords = list(geom.coords)
    (x0, y0) = coords[0][:2]
    (x1, y1) = coords[-1][:2]
    d = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    # Transformer coordonnées en indices raster
    
    row0, col0 = xy_to_rowcol(x0, y0, out_meta)
    row1, col1 = xy_to_rowcol(x1, y1, out_meta)

    # Vérifier les bornes
    h, w = mosaic.shape[1], mosaic.shape[2]

    alt0 = safe_get(row0, col0, h, w, mosaic, out_meta)
    alt1 = safe_get(row1, col1, h, w, mosaic, out_meta)
    diff = None
    slope = None
    if opt_abs:
        if alt0 is not None and alt1 is not None:
            diff = abs(alt1 - alt0)
            slope = (diff / d) * 100 if d != 0 else None
    else:
        if alt0 is not None and alt1 is not None:
            diff = alt1 - alt0
            slope = (diff / d) * 100 if d != 0 else None
    return alt0, alt1, diff, slope

def compute_altitude_diff_for_gdf(gdf, mosaic, out_meta, opt_abs=True):
    alt_debut = []
    alt_fin = []
    alt_diff = []
    slopes = []
    for idx, row in gdf.iterrows():
        # Si la nature est "escalier" ou position_par_rapport_au_sol != 0, on met None
        if str(row.get('nature', '')).strip().lower() == "escalier" or str(row.get('position_par_rapport_au_sol', '0')) != "0":
            a0, a1, diff, slope = None, None, None, 0
        else:
            a0, a1, diff, slope = get_altitude_difference_from_rgalti(row.geometry, mosaic, out_meta, opt_abs=opt_abs)
        alt_debut.append(a0)
        alt_fin.append(a1)
        alt_diff.append(diff)
        slopes.append(slope)
    gdf['altitude_debut_rgalti'] = alt_debut
    gdf['altitude_fin_rgalti'] = alt_fin
    gdf['diff_altitude_rgalti'] = alt_diff
    gdf['slope_rgalti'] = slopes
    return gdf

def decouper_troncons(gdf, longueur_max, longueur_min=5):
    """
    Découpe les objets LineString d'un GeoDataFrame en segments de longueur maximale spécifiée.

    Pour chaque géométrie de type LineString dans le GeoDataFrame `gdf`, la fonction découpe la ligne en segments de longueur maximale `longueur_max`. 
    Les segments dont la longueur est inférieure à `longueur_min` sont ignorés. 
    Les autres types de géométrie sont conservés sans modification.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame contenant les géométries à découper.
        longueur_max (float): Longueur maximale des segments découpés.
        longueur_min (float, optionnel): Longueur minimale des segments à conserver (par défaut 5).

    Returns:
        gpd.GeoDataFrame: Un nouveau GeoDataFrame contenant les segments découpés et les autres géométries d'origine, 
        avec une colonne supplémentaire 'length' indiquant la longueur de chaque géométrie.
    """
    lignes = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if isinstance(geom, LineString):
            longueur = geom.length
            if longueur <= longueur_max:
                lignes.append(row)
            else:
                # Découper la ligne en segments de longueur_max
                points = [geom.interpolate(d) for d in np.arange(0, longueur, longueur_max)]
                points.append(geom.interpolate(longueur))
                for i in range(len(points)-1):
                    segment = LineString([points[i], points[i+1]])
                    if segment.length >= longueur_min:  # <-- Ajout du filtre
                        new_row = row.copy()
                        new_row.geometry = segment
                        lignes.append(new_row)
        else:
            lignes.append(row)
    gdf = gpd.GeoDataFrame(lignes, columns=gdf.columns, crs=gdf.crs)
    gdf['length'] = gdf.geometry.length
    return gdf

def calc_moy_pond_rgalti(row):
    """
    Calcule la moyenne pondérée des pentes ('slope_rgalti') en fonction des longueurs associées ('length') pour une ligne de DataFrame.

    Si 'slope_rgalti' et 'length' sont des listes de même longueur, la fonction retourne la moyenne pondérée des pentes, pondérée par les longueurs.
    Si la somme des longueurs est nulle, retourne np.nan.
    Si 'slope_rgalti' n'est pas une liste, tente de le convertir en float.
    En cas d'échec de conversion, retourne np.nan.

    Paramètres
    ----------
    row : pandas.Series
        Ligne d'un DataFrame contenant les clés 'slope_rgalti' et 'length'.

    Retourne
    --------
    float
        Moyenne pondérée des pentes, ou np.nan si les données sont invalides.
    """
    slopes = row['slope_rgalti']
    lengths = row['length']
    if isinstance(slopes, list) and isinstance(lengths, list) and len(slopes) == len(lengths):
        slopes_arr = np.array(slopes, dtype=np.float64)
        lengths_arr = np.array(lengths, dtype=np.float64)
        if lengths_arr.sum() == 0:
            return np.nan
        return np.average(slopes_arr, weights=lengths_arr)
    try:
        return float(slopes)
    except Exception:
        return np.nan
    
def calc_moy_pond_pente_1(row):
    slopes = row['pente_1']
    lengths = row['length']
    if isinstance(slopes, list) and isinstance(lengths, list) and len(slopes) == len(lengths):
        slopes_arr = np.array(slopes, dtype=np.float64)
        lengths_arr = np.array(lengths, dtype=np.float64)
        if lengths_arr.sum() == 0:
            return np.nan
        return np.average(slopes_arr, weights=lengths_arr)
    try:
        return float(slopes)
    except Exception:
        return np.nan    

def calc_moy_pond_pente_2(row):
    slopes = row['pente_2']
    lengths = row['length']
    if isinstance(slopes, list) and isinstance(lengths, list) and len(slopes) == len(lengths):
        slopes_arr = np.array(slopes, dtype=np.float64)
        lengths_arr = np.array(lengths, dtype=np.float64)
        if lengths_arr.sum() == 0:
            return np.nan
        return np.average(slopes_arr, weights=lengths_arr)
    try:
        return float(slopes)
    except Exception:
        return np.nan  
    
def moyenne_ponderee_slope(gdf, slope_col='slope_rgalti'):
    """
    Calcule la moyenne pondérée de la pente (slope_rgalti) par la longueur (length)
    pour chaque ligne du GeoDataFrame. Si slope_rgalti ou length est une liste,
    la moyenne pondérée est calculée, sinon la valeur unique est retournée.

    Ajoute une colonne 'slope_rgalti_moy_pond' au GeoDataFrame.

    Paramètres :
        gdf (gpd.GeoDataFrame) : Le GeoDataFrame à traiter.

    Retour :
        gpd.GeoDataFrame : Le GeoDataFrame avec la colonne ajoutée.
    """
    gdf = gdf.copy()
    if slope_col == 'slope_rgalti':
        gdf['slope_rgalti'] = gdf.apply(calc_moy_pond_rgalti, axis=1)
    elif slope_col == 'pente':
        gdf['pente_1'] = gdf.apply(calc_moy_pond_pente_1, axis=1)
        gdf['pente_2'] = gdf.apply(calc_moy_pond_pente_2, axis=1)
    return gdf

def ajouter_colonnes_sens(gdf):
    """
    Ajoute trois colonnes booléennes 'sens_1', 'sens_2', 'sens_3' à un GeoDataFrame selon la colonne 'sens_de_circulation'.
    - 'sens_1' : True si 'sens_de_circulation' == 'Sens direct'
    - 'sens_2' : True si 'sens_de_circulation' == 'Sens inverse'
    - 'sens_3' : True si 'sens_de_circulation' == 'Double sens' ou 'Sans objet'
    Les valeurs sont False par défaut.

    Paramètres :
        gdf (gpd.GeoDataFrame) : Le GeoDataFrame à traiter.

    Retour :
        gpd.GeoDataFrame : Le GeoDataFrame avec les colonnes ajoutées.
    """
    gdf = gdf.copy()
    gdf['sens_1'] = gdf['sens_de_circulation'] == 'Sens direct'
    gdf['sens_2'] = gdf['sens_de_circulation'] == 'Sens inverse'
    gdf['sens_3'] = gdf['sens_de_circulation'].isin(['Double sens', 'Sans objet'])
    return gdf

def calc_pente_1(row):
        if row['sens_3']:
            return row['slope_rgalti']
        elif row['sens_1']:
            return row['slope_rgalti']
        elif row['sens_2']:
            return None
        return None

def calc_pente_2(row):
        if row['sens_3']:
            try:
                return -float(row['slope_rgalti'])
            except Exception:
                return None
        elif row['sens_2']:
            return -float(row['slope_rgalti'])
        elif row['sens_1']:
            return None
        return None
    
def ajouter_colonnes_pente(gdf):
    """
    Ajoute deux colonnes 'pente_1' et 'pente_2' selon les colonnes 'sens_1', 'sens_2', 'sens_3' et 'slope_rgalti'.
    - Si 'sens_1' est True : 'pente_1' = slope_rgalti, 'pente_2' = None
    - Si 'sens_2' est True : 'pente_1' = None, 'pente_2' = slope_rgalti
    - Si 'sens_3' est True : 'pente_1' = slope_rgalti, 'pente_2' = -slope_rgalti

    Paramètres :
        gdf (gpd.GeoDataFrame) : Le GeoDataFrame à traiter.

    Retour :
        gpd.GeoDataFrame : Le GeoDataFrame avec les colonnes ajoutées.
    """
    gdf = gdf.copy()
    gdf['pente_1'] = gdf.apply(calc_pente_1, axis=1)
    gdf['pente_2'] = gdf.apply(calc_pente_2, axis=1)
    return gdf

def apply_slope (gdf, polygon_city):
    gdf = remplacer_nan_colonne(gdf, 'nombre_de_voies', 1)
    gdf = remplacer_nan_colonne(gdf, 'largeur_de_chaussee', 0)
    gdf = decouper_troncons(gdf, 10)
    mosaic, out_meta = import_and_merge_asc_for_city(polygon_city)
    gdf = compute_altitude_diff_for_gdf(gdf, mosaic, out_meta, opt_abs = False)
    gdf = ajouter_colonnes_sens(gdf)
    gdf = ajouter_colonnes_pente(gdf)
    gdf = fusionner_par_attribut(gdf, 'cleabs')
    gdf = moyenne_ponderee_slope(gdf, 'pente')
    gdf = fix_gdf(gdf)
    return gdf


#green and blue space
def percentage_green_space (gdf_routes, gdf_parc, gdf_green, gdf_blue, buffer_distance=10):
    """
    Calcule le pourcentage de recouvrement des espaces verts et bleus autour des routes.
    Cette fonction prend en entrée des GeoDataFrames représentant les routes, les parcs, les espaces verts et les espaces bleus.
    Pour chaque tronçon de route, elle crée un buffer (zone tampon) d'une distance spécifiée, puis calcule le pourcentage de la surface
    de ce buffer qui intersecte les espaces verts et bleus. Si la route est souterraine (définie par l'attribut 'position_par_rapport_au_sol'),
    le pourcentage est fixé à 0. Si la route traverse un parc, le pourcentage d'espace vert est fixé à 100.
    Args:
        gdf_routes (GeoDataFrame): GeoDataFrame contenant les géométries des routes et l'attribut 'position_par_rapport_au_sol'.
        gdf_parc (GeoDataFrame): GeoDataFrame des parcs.
        gdf_green (GeoDataFrame): GeoDataFrame des espaces verts.
        gdf_blue (GeoDataFrame): GeoDataFrame des espaces bleus (plans d'eau, rivières, etc.).
        buffer_distance (float, optional): Distance du buffer autour de chaque route, en mètres. Par défaut à 10.
    Returns:
        GeoDataFrame: Le GeoDataFrame d'entrée des routes, enrichi de deux colonnes :
            - 'green_overlap_percent' : pourcentage de recouvrement du buffer par des espaces verts.
            - 'blue_overlap_percent' : pourcentage de recouvrement du buffer par des espaces bleus.
    """
    gdf_routes = gdf_routes.to_crs('EPSG:2154')
    gdf_green = gdf_green.to_crs('EPSG:2154')
    gdf_blue = gdf_blue.to_crs('EPSG:2154')
    gdf_parc = gdf_parc.to_crs('EPSG:2154')

    # Creation of a buffer around each edge
    gdf_routes['buffer'] = gdf_routes['geometry'].buffer(buffer_distance)

    # Calcul of the overlap between buffers and green spaces
    overlap_percentages_green = []
    overlap_percentages_blue = []
    for idx, row in gdf_routes.iterrows():
        buffer_geom = row['buffer']
        geom = row['geometry']
        buffer_area = buffer_geom.area
        if row['position_par_rapport_au_sol'] in ['-1', '-2', '-3', '-4', '-5']:
            overlap_percentages_green.append(0)
        elif geom.intersects(gdf_parc.unary_union):
            overlap_percentages_green.append(100)
        else:
            intersection_green = gdf_green.geometry.intersection(buffer_geom)
            intersection_area_green = intersection_green.area.sum()
            overlap_percentage_green = (intersection_area_green / buffer_area) * 100 if buffer_area > 0 else 0
            overlap_percentages_green.append(overlap_percentage_green)
        
        if row['position_par_rapport_au_sol'] in ['-1', '-2', '-3', '-4', '-5']:
            overlap_percentages_blue.append(0)
        else:
            intersection_blue = gdf_blue.geometry.intersection(buffer_geom)
            intersection_area_blue = intersection_blue.area.sum()
            overlap_percentage_blue = (intersection_area_blue / buffer_area) * 100 if buffer_area > 0 else 0
            overlap_percentages_blue.append(overlap_percentage_blue)
    
    # Ajouter les pourcentages de superposition au GeoDataFrame
    gdf_routes['green_overlap_percent'] = overlap_percentages_green
    gdf_routes['blue_overlap_percent'] = overlap_percentages_blue

    return gdf_routes

def apply_green_and_blue (gdf, polygon_city, gdf_zones, buffer_distance):
    """
    Applique le calcul des espaces verts et bleus à un GeoDataFrame en fonction d'une zone urbaine donnée.

    Cette fonction lit les données géographiques des espaces naturels et des plans d'eau dans la zone d'intérêt,
    filtre les entités pertinentes (parcs, espaces verts, plans d'eau au niveau du sol) et calcule le pourcentage
    d'espaces verts et bleus autour des entités du GeoDataFrame d'entrée, selon une distance de buffer spécifiée.

    Args:
        gdf (GeoDataFrame): GeoDataFrame contenant les entités à enrichir avec les pourcentages d'espaces verts et bleus.
        polygon_city (Polygon): Polygone représentant la zone urbaine d'intérêt (ville).
        gdf_zones (GeoDataFrame): GeoDataFrame contenant les zones détaillées (ex: parcs).
        buffer_distance (float): Distance du buffer (en unités de projection) autour de chaque entité pour le calcul des pourcentages.

    Returns:
        GeoDataFrame: Le GeoDataFrame d'entrée enrichi avec les pourcentages d'espaces verts et bleus.
    """
    bbox = polygon_city.bounds
    gdf_eau = gpd.read_file(filepath_bdtopo_eau, bbox=bbox)
    gdf_ocean = gpd.read_file(filepath_ocean, bbox=bbox)
    gdf_ocean['cleabs'] = 99
    gdf_vert = gpd.read_file(filepath_bdtopo_nature, bbox=bbox)
    gdf_eau = gdf_eau[gdf_eau['position_par_rapport_au_sol'] == '0']
    gdf_eau = gdf_eau[gdf_eau.intersects(polygon_city)]
    gdf_eau = gpd.GeoDataFrame(pd.concat([gdf_eau, gdf_ocean], ignore_index=True), crs=gdf_eau.crs)
    gdf_parc = gdf_zones[gdf_zones['nature_detaillee'] == 'Parc']
    gdf_parc = gdf_parc[gdf_parc.intersects(polygon_city)]
    gdf_vert = gdf_vert[gdf_vert.intersects(polygon_city)]
    gdf = percentage_green_space(gdf, gdf_parc, gdf_vert, gdf_eau, buffer_distance=buffer_distance)
    return gdf, gdf_vert, gdf_parc, gdf_eau

#Main function to import BDTOPO data for a specific city
def importbdtopo(city,
                filepath_bdtopo=filepath_bdtopo, 
                filepath_bdadmin=filepath_bdadmin,
                filepath_bdtopo_za=filepath_bdtopo_za,
                filepath_bdtopo_ci=filepath_bdtopo_ci,
                buffer_distance=10,
                slope=True,
                green_and_blue=True):
    """
    Importe et prépare les données BDTOPO pour une commune donnée.

    Cette fonction charge les tronçons de routes, zones d'activités, cimetières et, en option, calcule les pentes et les pourcentages d'espaces verts/bleus autour des routes, pour une commune spécifiée par son nom.

    Paramètres
    ----------
    city : str
        Nom de la commune à traiter (doit correspondre à la colonne "NOM" du fichier BDAdmin).
    filepath_bdtopo : str, optionnel
        Chemin du fichier BDTOPO contenant les tronçons de routes.
    filepath_bdadmin : str, optionnel
        Chemin du fichier BDAdmin contenant les limites administratives des communes.
    filepath_bdtopo_za : str, optionnel
        Chemin du fichier BDTOPO contenant les zones d'activités ou d'intérêt.
    filepath_bdtopo_ci : str, optionnel
        Chemin du fichier BDTOPO contenant les cimetières.
    buffer_distance : float, optionnel
        Distance du buffer (en mètres) autour des routes pour le calcul des espaces verts/bleus (par défaut 10).
    slope : bool, optionnel
        Si True, calcule les pentes à partir du MNT RGALTI (par défaut True).
    green_and_blue : bool, optionnel
        Si True, calcule les pourcentages d'espaces verts et bleus autour des routes (par défaut True).

    Retourne
    -------
    gdf_routes : GeoDataFrame
        Tronçons de routes de la commune, enrichis avec les attributs calculés (pente, espaces verts/bleus, etc.).
    gdf_zones_et_cimetieres : GeoDataFrame
        Zones d'activités et cimetières, avec indication des zones exclues.
    gdf_vert : GeoDataFrame ou np.nan
        Espaces verts de la commune (ou np.nan si green_and_blue=False).
    gdf_parc : GeoDataFrame ou np.nan
        Parcs de la commune (ou np.nan si green_and_blue=False).
    gdf_eau : GeoDataFrame ou np.nan
        Espaces bleus (hydrographie) de la commune (ou np.nan si green_and_blue=False).
    polygon_city : shapely Polygon
        Polygone représentant l'emprise spatiale de la commune.

    """
    
    #importation et découpe de BDTOPO (tronçons de routes) en fonction de la commune 
    gdf_commune = gpd.read_file(filepath_bdadmin) #importation de BDAdmin
    gdf_commune = gdf_commune[gdf_commune["NOM"] == city] #Recherche de la commune dans BDAdmin
    polygon_city = gdf_commune.geometry.iloc[0] #Emprise spatiale de la commune (sous forme de polygon)
    bbox_city = tuple(gdf_commune.total_bounds) #Emprise spatiale de la commune (coordonées)
    gdf_routes = gpd.read_file(filepath_bdtopo, bbox=bbox_city) #importation de BDTOPO selon les coordonées de la commune
    gdf_routes = gdf_routes[gdf_routes.intersects(polygon_city)] #découpe de BDTOPO selon l'emprise spatiale de la commune
    gdf_routes = gdf_routes[~gdf_routes['fictif'].astype(bool)]
    
    #importation de BDTOPO et découpe (zones d'activité et cimetiéres)
    
    gdf_zones = gpd.read_file(filepath_bdtopo_za, bbox=bbox_city)
    gdf_zones = gdf_zones[gdf_zones.intersects(polygon_city)]
    gdf_cimetieres = gpd.read_file(filepath_bdtopo_ci, bbox=bbox_city)
    gdf_cimetieres = gdf_cimetieres[gdf_cimetieres.intersects(polygon_city)]
    
    cols_to_keep = [col for col in gdf_cimetieres.columns if col in gdf_zones.columns] #fusion de gdf_zones et gdf_cimetieres
    gdf_cimetieres = gdf_cimetieres[cols_to_keep]
    gdf_cimetieres['nature'] = "Cimetière"
    gdf_cimetieres['categorie'] = "Cimetière"
    gdf_zones_et_cimetieres = pd.concat([gdf_zones, gdf_cimetieres], ignore_index=True)
    
    gdf_zones_et_cimetieres['zone_exclue'] = gdf_zones_et_cimetieres.apply(set_zones_exclues, axis=1) #détermination des zones considérée comme restraintes
    
    
    #application des amenagements
    apply_amenagment(gdf_routes)
    
    #pentes
    if slope :
        gdf_routes = apply_slope(gdf_routes, polygon_city=polygon_city)
    #green and blue space
    if green_and_blue :
        gdf_routes, gdf_vert, gdf_parc, gdf_eau = apply_green_and_blue(gdf_routes, 
                                                       polygon_city=polygon_city, 
                                                       gdf_zones=gdf_zones,
                                                       buffer_distance=buffer_distance)
    if not green_and_blue :
        gdf_vert = np.nan
        gdf_parc = np.nan
        gdf_eau = np.nan
    
    cols_to_keep = [col for col in list_colones_transport if col in gdf_routes.columns] #suppresion des données inutiles
    gdf_routes = gdf_routes[cols_to_keep]
    
    return gdf_routes, gdf_zones_et_cimetieres, gdf_vert, gdf_parc, gdf_eau, polygon_city