#file paths
base_filepath = 'data/IGN/'
filepath_bdtopo = base_filepath + 'BDTOPO_Express_v3.4/BDTOPO-EXPRESS_3-4__GPKG_LAMB93_FXX_2025-06-01/BDTOPO-EXPRESS/1_DONNEES_LIVRAISON_2025-06-00002/BDT_3-4_EXPRESS_GPKG_LAMB93_FXX_ED2025-06-01/TRANSPORT/troncon_de_route.gpkg'
filepath_bdadmin = base_filepath + 'ADMIN-EXPRESS_3-2__SHP_LAMB93_FXX_2024-03-25/ADMIN-EXPRESS_3-2__SHP_LAMB93_FXX_2024-03-25/ADMIN-EXPRESS/1_DONNEES_LIVRAISON_2024-03-00237/ADE_3-2_SHP_LAMB93_FXX-ED2024-03-25/COMMUNE.shp'
filepath_bdtopo_za = base_filepath + 'BDTOPO_Express_v3.4/BDTOPO-EXPRESS_3-4__GPKG_LAMB93_FXX_2025-06-01/BDTOPO-EXPRESS/1_DONNEES_LIVRAISON_2025-06-00002/BDT_3-4_EXPRESS_GPKG_LAMB93_FXX_ED2025-06-01/SERVICES_ET_ACTIVITES/zone_d_activite_ou_d_interet.gpkg'
filepath_bdtopo_ci = base_filepath + 'BDTOPO_Express_v3.4/BDTOPO-EXPRESS_3-4__GPKG_LAMB93_FXX_2025-06-01/BDTOPO-EXPRESS/1_DONNEES_LIVRAISON_2025-06-00002/BDT_3-4_EXPRESS_GPKG_LAMB93_FXX_ED2025-06-01/BATI/cimetiere.gpkg'
filepath_bdtopo_eau = base_filepath + 'BDTOPO_Express_v3.4/BDTOPO-EXPRESS_3-4__GPKG_LAMB93_FXX_2025-06-01/BDTOPO-EXPRESS/1_DONNEES_LIVRAISON_2₀₂₅₋₀₆₋₀₀₀₀₂/BDT₃₋₄_EXPRESS_GPKG_LAMB93_FXX_ED₂₀₂₅₋₀₆₋₀₁/HYDROGRAPHIE/surface_hydrographique.gpkg'
filepath_bdtopo_nature = base_filepath + 'BDTOPO_Express_v3.4/BDTOPO-EXPRESS_3-4__GPKG_LAMB93_FXX_2025-06-01/BDTOPO-EXPRESS/1_DONNEES_LIVRAISON_2025-06-00002/BDT_3-4_EXPRESS_GPKG_LAMB93_FXX_ED2025-06-01/OCCUPATION_DU_SOL/zone_de_vegetation.gpkg'
filepath_ocean = base_filepath + 'ocean_mer_2.gpkg'


# Define dictionaries and lists for amenagement and restrictions
dict_categories_amenagement_cyclable = {0 : "Pas d'aménagement cyclable",
                                        1 : "Chaussée partagée aménagée",
                                        2 : "Aménagement non séparé partagé",
                                        3 : "Aménagement séparé partagé",
                                        4 : "Aménagement séparé non partagé"}

dict_nature_de_la_restriction = {"Aménagement mixte hors voie verte": 3,
                                "Chaussée à voie centrale banalisée": 1,
                                "Double sens cyclable non matérialisé": 0,
                                "Piste cyclable": 4,
                                "Vélorue" : 4,
                                "Voie verte": 3,
                                "Sentier" : 3}

dict_amenagment_cyclable = {"Aménagement mixte hors voie verte": 2,
                            "Bande cyclable": 1,
                            "Piste cyclable" : 4,
                            "Sentier" : 3}

list_zones_exclues_nature = ["Autre établissement d'enseignement",
                             "Autre service déconcentré de l'Etat",
                             "Camping",
                             "Camp militaire non clos",
                             "Capitainerie",
                             "Carrière",
                             "Caserne",
                             "Caserne de pompiers",
                             "Centrale électrique",
                             "Champ de tir",
                             "Collège",
                             "Complexe sportif couvert",
                             "Déchèterie",
                             "Ecomusée",
                             "Enceinte militaire",
                             "Enseignement primaire",
                             "Enseignement supérieur",
                             "Equipement de cyclisme",
                             "Etablissement extraterritorial",
                             "Etablissement hospitalier",
                             "Etablissement pénitentiaire",
                             "Etablissement thermal",
                             "Gendarmerie",
                             "Golf",
                             "Hippodrome",
                             "Hôpital",
                             "Hôtel de collectivité",
                             "Hôtel de département",
                             "Hôtel de région",
                             "Lycée",
                             "Marché",
                             "Mine",
                             "Ouvrage militaire",
                             "Parc de loisirs",
                             "Parc des expositions",
                             "Parc zoologique",
                             "Patinoire",
                             "Piscine",
                             "Police",
                             "Poste",
                             "Préfecture",
                             "Préfecture de région",
                             "Salle de spectacle ou conférence",
                             "Science",
                             "Siège d'EPCI",
                             "Station de pompage",
                             "Station d'épuration",
                             "Surveillance maritime",
                             "Université",
                             "Usine",
                             "Usine de production d'eau potable",
                             "Cimetière"]

list_zones_exclues_nature_detaillee = ["Jardin botanique",
                                       "Jardin familiaux",
                                       "Parc"]

list_colones_transport = ['cleabs',
                          'geometry',
                          'nature',
                          'nom_collaboratif_gauche',
                          'nom_collaboratif_droite',
                          'importance',
                          'position_par_rapport_au_sol',
                          'nombre_de_voies',
                          'largeur_de_chaussee',
                          'sens_de_circulation',
                          'reserve_aux_bus',
                          'vitesse_moyenne_vl',
                          'acces_vehicule_leger',
                          'acces_pieton',
                          'nature_de_la_restriction',
                          'sens_amenagement_cyclable_gauche','sens_amenagement_cyclable_droit',
                          'amenagement_cyclable_gauche',
                          'amenagement_cyclable_droit',
                          'presence_amenagement',
                          'nom_amenagement',
                          'nom_amenagement_droit',
                          'nom_amenagement_gauche',
                          'length',
                          'sens_1',
                          'sens_2',
                          'sens_3',
                          'pente_1',
                          'pente_2',
                          'green_overlap_percent',
                          'blue_overlap_percent']
