import tkinter as tk
from tkinter import ttk, messagebox
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils_benchmark import *


class MultiGroupFilterPanel:
    """Représente un panneau de filtres avec sélection multiple par cases à cocher pour chaque variable"""

    def __init__(self, parent_container, group_id, df_complet, on_change_callback):
        self.group_id = group_id
        self.df_complet = df_complet
        self.on_change_callback = on_change_callback

        # Cadre principal du groupe
        self.frame = ttk.LabelFrame(parent_container, text=f" Groupe {group_id} ", padding=8)
        self.frame.pack(fill=tk.X, padx=5, pady=5)

        # Nom personnalisable pour la légende du graphique
        name_frame = ttk.Frame(self.frame)
        name_frame.pack(fill=tk.X, pady=2)
        ttk.Label(name_frame, text="Nom :", width=8).pack(side=tk.LEFT)
        self.name_var = tk.StringVar(value=f"Groupe {group_id}")
        self.name_entry = ttk.Entry(name_frame, textvariable=self.name_var, width=20)
        self.name_entry.pack(side=tk.LEFT, padx=5)
        self.name_var.trace_add("write", lambda *args: self.on_change_callback())

        # Conteneur pour les filtres de ce groupe
        filters_container = ttk.Frame(self.frame)
        filters_container.pack(fill=tk.X, pady=5)

        # Regroupement intelligent des colonnes One-Hot par variable d'origine
        self.variable_mapping = {}
        cat_base_names = ['age', 'gender', 'electric_bike', 'speed', 'slope', 'green', 'type',
                          'bike_use_frequency', 'bike_ownership', 'nbr_lane', 'id_personne', 'id_photo']

        for col in self.df_complet.columns:
            if col in ['note', 'original_id_photo']:
                continue
            matched = False
            for base in cat_base_names:
                if col.startswith(base + '_'):
                    val = col[len(base) + 1:]
                    if base not in self.variable_mapping:
                        self.variable_mapping[base] = []
                    self.variable_mapping[base].append((val, col))
                    matched = True
                    break
            if not matched:
                self.variable_mapping[col] = [(col, col)]

        # Dictionnaire pour stocker les variables booléennes de chaque case à cocher
        self.filter_vars = {}

        # Génération d'une ligne de cases à cocher pour chaque variable d'origine (sauf id_photo géré globalement)
        for var_name in sorted(self.variable_mapping.keys()):
            if var_name == 'id_photo':
                continue

            row_f = ttk.Frame(filters_container, padding=2)
            row_f.pack(fill=tk.X, anchor="w", pady=2)

            lbl = ttk.Label(row_f, text=f"{var_name}:", width=16, anchor="w")
            lbl.pack(side=tk.LEFT, anchor="n")

            vals_frame = ttk.Frame(row_f)
            vals_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

            possible_values = [val for val, _ in self.variable_mapping[var_name]]
            try:
                possible_values = sorted(possible_values, key=lambda x: float(x))
            except ValueError:
                possible_values = sorted(possible_values)

            self.filter_vars[var_name] = {}

            chk_container = ttk.Frame(vals_frame)
            chk_container.pack(anchor="w", fill=tk.X)

            for val in possible_values:
                b_var = tk.BooleanVar(value=False)
                self.filter_vars[var_name][val] = b_var

                chk = ttk.Checkbutton(
                    chk_container,
                    text=str(val),
                    variable=b_var,
                    command=self.on_change_callback
                )
                chk.pack(side=tk.LEFT, padx=3)

    def get_filtered_notes(self, global_photo_val):
        """Applique les filtres du groupe ainsi que le filtre global de photo basé sur l'ID d'origine"""
        mask = pd.Series(True, index=self.df_complet.index)

        # Filtre global sur la photo (via la colonne originale id_photo de 1 à 17)
        if global_photo_val != "Toutes":
            mask = mask & (self.df_complet['original_id_photo'] == int(global_photo_val))

        # Filtres internes au groupe (cases à cocher)
        for var_name, vals_dict in self.filter_vars.items():
            selected_vals = [val for val, b_var in vals_dict.items() if b_var.get()]

            if selected_vals:
                sub_mask = pd.Series(False, index=self.df_complet.index)
                cols_for_var = self.variable_mapping[var_name]

                for val, col_name in cols_for_var:
                    if str(val) in [str(sv) for sv in selected_vals]:
                        if col_name in self.df_complet.columns:
                            sub_mask = sub_mask | (self.df_complet[col_name] == 1)

                mask = mask & sub_mask

        df_filtered = self.df_complet[mask]
        return df_filtered['note'], self.name_var.get()


class InteractiveHistogramApp:
    def __init__(self, root, base_csv, personnes_csv, photos_csv=None):
        self.root = root
        self.root.title("Comparateur Interactif d'Histogrammes et Statistiques")
        self.root.geometry("1400x950")

        # Dossier par défaut pour les sauvegardes automatiques
        self.output_dir = os.path.join("outputs", "survey", "data_explorer_output")
        os.makedirs(self.output_dir, exist_ok=True)

        # Chargement des données brutes pour récupérer les vrais id_photo (1 à 17)
        try:
            df_all_raw = import_and_merge_data_base(base_csv, personnes_csv, photos_csv)
            df_all_raw = df_all_raw[df_all_raw['job'] == 6]
            df_all_raw = df_all_raw[df_all_raw['postal_code'].between(69000, 70000, inclusive='neither')]
            original_photo_series = df_all_raw['id_photo'].reset_index(drop=True)
        except Exception as e:
            original_photo_series = None

        # Chargement des données via la fonction demandée
        try:
            df_enc, encoded_cols = load_and_preprocess_data(base_csv, personnes_csv, photos_csv, drop=None, people=True)
            self.X = df_enc[encoded_cols].copy()
            self.X['id_personne'] = df_enc['id_personne']
            self.X['id_photo'] = df_enc['id_photo']
            self.y = df_enc["note"].astype(int)
        except Exception as e:
            messagebox.showerror("Erreur de chargement", f"Impossible de charger les données :\n{e}")
            root.destroy()
            return

        self.df_complet = self.X.copy()
        self.df_complet['note'] = self.y.values + 1  # Échelle originale des notes

        # On réassocie les vrais numéros de photos (1 à 17) alignés avec le DataFrame filtré
        if original_photo_series is not None and len(original_photo_series) == len(self.df_complet):
            self.df_complet['original_id_photo'] = original_photo_series.values
        else:
            # Fallback si les index diffèrent
            self.df_complet['original_id_photo'] = df_all_raw['id_photo'].values[:len(self.df_complet)]

        self.groups = []
        self.group_counter = 0

        self.create_widgets()

        # Création par défaut de deux groupes pour démarrer
        self.add_group()
        self.add_group()

    def create_widgets(self):
        vertical_pane = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        vertical_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- PARTIE HAUTE : Filtres et Graphique ---
        top_pane = ttk.PanedWindow(vertical_pane, orient=tk.HORIZONTAL)
        vertical_pane.add(top_pane, weight=3)

        # 1. Panneau de gauche : Gestion des groupes et filtres
        left_outer_frame = ttk.Frame(top_pane)
        top_pane.add(left_outer_frame, weight=2)

        btn_frame = ttk.Frame(left_outer_frame, padding=5)
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text="+ Ajouter un groupe", command=self.add_group).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Supprimer le dernier", command=self.remove_group).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Réinitialiser tout", command=self.reset_all_groups).pack(side=tk.LEFT, padx=2)

        # --- MENU DÉROULANT GLOBAL POUR LA PHOTO ---
        photo_frame = ttk.LabelFrame(left_outer_frame, text=" Sélection globale de la photo ", padding=8)
        photo_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(photo_frame, text="Photo :").pack(side=tk.LEFT, padx=5)

        photo_choices = ["Toutes"]
        if 'original_id_photo' in self.df_complet.columns:
            photo_ids = sorted(self.df_complet['original_id_photo'].unique())
            photo_choices += [str(int(p)) for p in photo_ids]

        self.photo_var = tk.StringVar(value="Toutes")
        self.photo_dropdown = ttk.Combobox(
            photo_frame,
            textvariable=self.photo_var,
            values=photo_choices,
            state="readonly",
            width=15
        )
        self.photo_dropdown.pack(side=tk.LEFT, padx=5)
        self.photo_dropdown.bind("<<ComboboxSelected>>", lambda event: self.update_all())

        # Options (pourcentage et bouton d'enregistrement direct)
        options_frame = ttk.Frame(left_outer_frame, padding=5)
        options_frame.pack(fill=tk.X, padx=5)

        self.percentage_var = tk.BooleanVar(value=False)
        chk_percentage = ttk.Checkbutton(
            options_frame,
            text="Afficher en pourcentage (%)",
            variable=self.percentage_var,
            command=self.update_all
        )
        chk_percentage.pack(side=tk.LEFT, anchor="w")

        btn_save = ttk.Button(options_frame, text="Enregistrer", command=self.save_plot_automatic)
        btn_save.pack(side=tk.RIGHT, padx=5)

        # Zone scrollable pour les filtres
        canvas_frame = ttk.LabelFrame(left_outer_frame, text=" Groupes de Comparaison ", padding=5)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.canvas = tk.Canvas(canvas_frame, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_groups_frame = ttk.Frame(self.canvas)

        self.scrollable_groups_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_groups_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # 2. Panneau de droite : Graphique Matplotlib
        right_frame = ttk.LabelFrame(top_pane, text=" Superposition des Histogrammes ", padding=10)
        top_pane.add(right_frame, weight=3)

        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.canvas_matplotlib = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas_matplotlib.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- PARTIE BASSE : Tableau statistique récapitulatif ---
        bottom_frame = ttk.LabelFrame(vertical_pane, text=" Tableau Statistique Comparatif ", padding=10)
        vertical_pane.add(bottom_frame, weight=1)

        columns = ("groupe", "count", "mean", "std", "median", "min", "max")
        self.stats_table = ttk.Treeview(bottom_frame, columns=columns, show="headings", height=4)

        self.stats_table.heading("groupe", text="Groupe")
        self.stats_table.heading("count", text="Effectif (n)")
        self.stats_table.heading("mean", text="Moyenne")
        self.stats_table.heading("std", text="Écart-type")
        self.stats_table.heading("median", text="Médiane")
        self.stats_table.heading("min", text="Min")
        self.stats_table.heading("max", text="Max")

        for col in columns:
            self.stats_table.column(col, anchor="center", width=120)

        scrollbar_table = ttk.Scrollbar(bottom_frame, orient="vertical", command=self.stats_table.yview)
        self.stats_table.configure(yscrollcommand=scrollbar_table.set)

        self.stats_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_table.pack(side=tk.RIGHT, fill=tk.Y)

    def add_group(self):
        self.group_counter += 1
        group = MultiGroupFilterPanel(
            self.scrollable_groups_frame,
            self.group_counter,
            self.df_complet,
            self.update_all
        )
        self.groups.append(group)
        self.update_all()

    def remove_group(self):
        if len(self.groups) > 1:
            group = self.groups.pop()
            group.frame.destroy()
            self.update_all()
        else:
            messagebox.showwarning("Attention", "Vous devez conserver au moins un groupe.")

    def reset_all_groups(self):
        self.photo_var.set("Toutes")
        for group in self.groups:
            for vals_dict in group.filter_vars.values():
                for b_var in vals_dict.values():
                    b_var.set(False)
        self.update_all()

    def update_all(self, *args):
        self.update_plot()
        self.update_stats_table()

    def save_plot_automatic(self):
        global_photo = self.photo_var.get()
        group_names = [g.name_var.get().strip().replace(" ", "_") for g in self.groups]
        groups_str = "_vs_".join(group_names)

        mode_str = "pct" if self.percentage_var.get() else "count"
        photo_str = f"photo_{global_photo}" if global_photo != "Toutes" else "toutes_photos"

        filename = f"hist_{photo_str}_{groups_str}_{mode_str}.png"
        file_path = os.path.join(self.output_dir, filename)

        try:
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Enregistrement réussi", f"Le graphique a été enregistré dans :\n{file_path}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'enregistrer l'image :\n{e}")

    def update_plot(self):
        self.ax.clear()

        data_to_plot = []
        labels = []
        is_percentage = self.percentage_var.get()
        global_photo = self.photo_var.get()

        for group in self.groups:
            notes, name = group.get_filtered_notes(global_photo)
            data_to_plot.append(notes)
            labels.append(f"{name} (n={len(notes)})")

        if any(len(d) > 0 for d in data_to_plot):
            all_notes_concat = pd.concat(data_to_plot) if data_to_plot else pd.Series([1, 5])
            min_val = int(all_notes_concat.min())
            max_val = int(all_notes_concat.max())
            bins = range(min_val, max_val + 2)

            weights = []
            if is_percentage:
                for notes in data_to_plot:
                    if len(notes) > 0:
                        weights.append(np.ones(len(notes)) * (100.0 / len(notes)))
                    else:
                        weights.append([])
                self.ax.hist(
                    data_to_plot,
                    bins=bins,
                    weights=weights,
                    align='left',
                    rwidth=0.85,
                    alpha=0.6,
                    label=labels,
                    edgecolor='black'
                )
                self.ax.set_ylabel("Pourcentage (%)", fontsize=11)
            else:
                self.ax.hist(
                    data_to_plot,
                    bins=bins,
                    align='left',
                    rwidth=0.85,
                    alpha=0.6,
                    label=labels,
                    edgecolor='black'
                )
                self.ax.set_ylabel("Nombre d'occurrences", fontsize=11)

            photo_title_str = f" (Photo : {global_photo})" if global_photo != "Toutes" else " (Toutes les photos)"
            self.ax.set_title(f"Comparaison des histogrammes de notes{photo_title_str}", fontsize=12, fontweight='bold')
            self.ax.set_xlabel("Notes", fontsize=11)
            self.ax.grid(axis='y', linestyle='--', alpha=0.6)
            self.ax.legend(loc='upper right', frameon=True)

            if len(all_notes_concat.unique()) > 0:
                self.ax.set_xticks(sorted(all_notes_concat.unique()))
        else:
            self.ax.text(0.5, 0.5, "Aucune donnée ne correspond\naux filtres sélectionnés",
                         horizontalalignment='center', verticalalignment='center',
                         transform=self.ax.transAxes, fontsize=12, color='red')
            self.ax.set_title("Comparaison des histogrammes (0 résultat)", fontsize=12, fontweight='bold')

        self.fig.tight_layout()
        self.canvas_matplotlib.draw()

    def update_stats_table(self):
        for row in self.stats_table.get_children():
            self.stats_table.delete(row)

        global_photo = self.photo_var.get()
        for group in self.groups:
            notes, name = group.get_filtered_notes(global_photo)
            count = len(notes)

            if count > 0:
                mean_val = f"{notes.mean():.2f}"
                std_val = f"{notes.std(ddof=1):.2f}" if count > 1 else "0.00"
                median_val = f"{notes.median():.2f}"
                min_val = f"{notes.min()}"
                max_val = f"{notes.max()}"
            else:
                mean_val = "-"
                std_val = "-"
                median_val = "-"
                min_val = "-"
                max_val = "-"

            self.stats_table.insert("", "end", values=(name, count, mean_val, std_val, median_val, min_val, max_val))

if __name__ == "__main__":
    base_csv = "data/survey/data-base.csv"
    personnes_csv = "data/survey/data-personnes.csv"
    photos_csv = "data/survey/data-photos.csv"

    root = tk.Tk()
    app = InteractiveHistogramApp(root, base_csv, personnes_csv, photos_csv)
    root.mainloop()