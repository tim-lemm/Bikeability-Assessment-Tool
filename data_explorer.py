import tkinter as tk
from tkinter import ttk, messagebox
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils_benchmark import *

# --- CONFIGURATION DES COULEURS DARK MODE (INTERFACE SEULEMENT) ---
BG_DARK = "#1e1e1e"
BG_PANEL = "#252526"
BG_WIDGET = "#2d2d2d"
FG_TEXT = "#e0e0e0"
ACCENT_COLOR = "#0e639c"
BORDER_COLOR = "#3f3f46"


class MultiGroupFilterPanel:
    """Représente un panneau de filtres avec sélection multiple par cases à cocher pour chaque variable"""

    def __init__(self, parent_container, group_id, df_complet, on_change_callback):
        self.group_id = group_id
        self.df_complet = df_complet
        self.on_change_callback = on_change_callback

        # Cadre principal du groupe
        self.frame = ttk.LabelFrame(parent_container, text=f" Group {group_id} ", padding=8)
        self.frame.pack(fill=tk.X, padx=5, pady=5)

        # Nom personnalisable pour la légende du graphique
        name_frame = ttk.Frame(self.frame)
        name_frame.pack(fill=tk.X, pady=2)
        ttk.Label(name_frame, text="Nom :", width=8).pack(side=tk.LEFT)
        self.name_var = tk.StringVar(value=f"Group {group_id}")
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

        if global_photo_val != "All":
            mask = mask & (self.df_complet['original_id_photo'] == int(global_photo_val))

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
        self.root.title("Data Explorer (Dark Mode + Moyennes)")
        self.root.geometry("1450x980")
        self.root.configure(bg=BG_DARK)

        # Configuration du style ttk pour le mode sombre de l'interface
        self.setup_styles()

        self.output_dir = os.path.join("outputs", "survey", "data_explorer_output")
        os.makedirs(self.output_dir, exist_ok=True)

        try:
            df_all_raw = import_and_merge_data_base(base_csv, personnes_csv, photos_csv)
            df_all_raw = df_all_raw[df_all_raw['job'] == 6]
            df_all_raw = df_all_raw[df_all_raw['postal_code'].between(69000, 70000, inclusive='neither')]
            original_photo_series = df_all_raw['id_photo'].reset_index(drop=True)
        except Exception as e:
            original_photo_series = None

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
        self.df_complet['note'] = self.y.values + 1

        if original_photo_series is not None and len(original_photo_series) == len(self.df_complet):
            self.df_complet['original_id_photo'] = original_photo_series.values
        else:
            self.df_complet['original_id_photo'] = df_all_raw['id_photo'].values[:len(self.df_complet)]

        self.groups = []
        self.group_counter = 0

        self.create_widgets()

        self.add_group()
        self.add_group()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        style.configure(".",
            background=BG_DARK,
            foreground=FG_TEXT,
            fieldbackground=BG_WIDGET,
            troughcolor=BG_PANEL,
            selectbackground=ACCENT_COLOR,
            selectforeground="#ffffff"
        )

        style.configure("TFrame", background=BG_DARK)
        style.configure("TLabelframe", background=BG_DARK, foreground=FG_TEXT, bordercolor=BORDER_COLOR)
        style.configure("TLabelframe.Label", background=BG_DARK, foreground=FG_TEXT, font=("Helvetica", 10, "bold"))

        style.configure("TLabel", background=BG_DARK, foreground=FG_TEXT)
        style.configure("TButton", background="#333333", foreground=FG_TEXT, bordercolor=BORDER_COLOR)
        style.map("TButton",
            background=[('active', '#3e3e42'), ('pressed', '#222222')],
            foreground=[('active', '#ffffff')]
        )

        style.configure("TCheckbutton", background=BG_DARK, foreground=FG_TEXT)
        style.map("TCheckbutton",
            background=[('active', BG_DARK)],
            foreground=[('active', FG_TEXT)]
        )

        style.configure("TCombobox", fieldbackground=BG_WIDGET, foreground=FG_TEXT, selectbackground=ACCENT_COLOR)

        # Style pour le Treeview (Tableau de stats)
        style.configure("Treeview",
            background="#252526",
            foreground=FG_TEXT,
            fieldbackground="#252526",
            bordercolor=BORDER_COLOR,
            rowheight=25
        )
        style.configure("Treeview.Heading",
            background="#333333",
            foreground=FG_TEXT,
            relief="flat"
        )
        style.map("Treeview.Heading",
            background=[('active', '#3e3e42')]
        )

    def create_widgets(self):
        vertical_pane = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        vertical_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        top_pane = ttk.PanedWindow(vertical_pane, orient=tk.HORIZONTAL)
        vertical_pane.add(top_pane, weight=3)

        # Panneau de gauche
        left_outer_frame = ttk.Frame(top_pane)
        top_pane.add(left_outer_frame, weight=2)

        btn_frame = ttk.Frame(left_outer_frame, padding=5)
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text="+ Add a group", command=self.add_group).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Delete last added group", command=self.remove_group).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Reset all", command=self.reset_all_groups).pack(side=tk.LEFT, padx=2)

        # Sélection de la photo
        photo_frame = ttk.LabelFrame(left_outer_frame, text=" Photo selection ", padding=8)
        photo_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(photo_frame, text="Photo :").pack(side=tk.LEFT, padx=5)

        photo_choices = ["All"]
        if 'original_id_photo' in self.df_complet.columns:
            photo_ids = sorted(self.df_complet['original_id_photo'].unique())
            photo_choices += [str(int(p)) for p in photo_ids]

        self.photo_var = tk.StringVar(value="All")
        self.photo_dropdown = ttk.Combobox(
            photo_frame,
            textvariable=self.photo_var,
            values=photo_choices,
            state="readonly",
            width=15
        )
        self.photo_dropdown.pack(side=tk.LEFT, padx=5)
        self.photo_dropdown.bind("<<ComboboxSelected>>", lambda event: self.update_all())

        # Options graphiques et bouton Enregistrer
        options_frame = ttk.LabelFrame(left_outer_frame, text="Options", padding=8)
        options_frame.pack(fill=tk.X, padx=5, pady=5)

        self.percentage_var = tk.BooleanVar(value=False)
        self.trend_var = tk.BooleanVar(value=False)
        self.boxplot_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(options_frame, text="Percentage (%)", variable=self.percentage_var,
                        command=self.update_all).pack(anchor="w", pady=2)
        ttk.Checkbutton(options_frame, text="Trend line", variable=self.trend_var,
                        command=self.update_all).pack(anchor="w", pady=2)
        ttk.Checkbutton(options_frame, text="Box-plot", variable=self.boxplot_var,
                        command=self.update_all).pack(anchor="w", pady=2)

        btn_save = ttk.Button(options_frame, text="Save plot", command=self.save_plot_automatic)
        btn_save.pack(anchor="e", pady=5)

        # Zone scrollable pour les filtres
        canvas_frame = ttk.LabelFrame(left_outer_frame, text=" Groups ", padding=5)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.canvas = tk.Canvas(canvas_frame, highlightthickness=0, bg=BG_DARK)
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

        # Panneau de droite : Graphique Matplotlib
        right_frame = ttk.LabelFrame(top_pane, padding=10)
        top_pane.add(right_frame, weight=3)

        self.fig, self.ax = plt.subplots(figsize=(7, 6))
        self.canvas_matplotlib = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas_matplotlib.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Partie basse : Tableau statistique
        bottom_frame = ttk.LabelFrame(vertical_pane, text=" Stats ", padding=10)
        vertical_pane.add(bottom_frame, weight=1)

        columns = ("group", "count", "mean", "std", "median", "min", "max")
        self.stats_table = ttk.Treeview(bottom_frame, columns=columns, show="headings", height=4)

        self.stats_table.heading("group", text="Group")
        self.stats_table.heading("count", text="Number (n)")
        self.stats_table.heading("mean", text="Mean")
        self.stats_table.heading("std", text="Standard deviation")
        self.stats_table.heading("median", text="Median")
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
        self.photo_var.set("All")
        self.percentage_var.set(False)
        self.trend_var.set(False)
        self.boxplot_var.set(False)
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
        trend_str = "_connected" if self.trend_var.get() else ""
        box_str = "_boxplot" if self.boxplot_var.get() else ""
        photo_str = f"photo_{global_photo}" if global_photo != "All" else "all_photos"

        filename = f"hist_{photo_str}_{groups_str}_{mode_str}{trend_str}{box_str}.png"
        file_path = os.path.join(self.output_dir, filename)

        try:
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Enregistrement réussi", f"Le graphique a été enregistré dans :\n{file_path}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'enregistrer l'image :\n{e}")

    def update_plot(self):
        self.ax.clear()
        self.fig.clf()

        data_to_plot = []
        labels = []
        is_percentage = self.percentage_var.get()
        show_trend = self.trend_var.get()
        show_boxplot = self.boxplot_var.get()
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

            if show_boxplot:
                ax_hist, ax_box = self.fig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            else:
                ax_hist = self.fig.subplots(1, 1)
                ax_box = None

            weights = []
            if is_percentage:
                for notes in data_to_plot:
                    if len(notes) > 0:
                        weights.append(np.ones(len(notes)) * (100.0 / len(notes)))
                    else:
                        weights.append([])
                n_list, bins_out, patches = ax_hist.hist(
                    data_to_plot,
                    bins=bins,
                    weights=weights,
                    align='left',
                    rwidth=0.85,
                    alpha=0.6,
                    label=labels,
                    edgecolor='black'
                )
                ax_hist.set_ylabel("Percentages (%)", fontsize=10)
            else:
                n_list, bins_out, patches = ax_hist.hist(
                    data_to_plot,
                    bins=bins,
                    align='left',
                    rwidth=0.85,
                    alpha=0.6,
                    label=labels,
                    edgecolor='black'
                )
                ax_hist.set_ylabel("Number", fontsize=10)

            # Extraction des couleurs des barres (couleurs classiques par défaut de Matplotlib)
            if len(data_to_plot) == 1:
                group_colors = [patches[0].get_facecolor()]
            else:
                group_colors = [patch_group[0].get_facecolor() for patch_group in patches]

            # Relier les sommets exactement au niveau des ticks de l'abscisse (bins_out[:-1])
            if show_trend:
                tick_positions = bins_out[:-1]
                if len(data_to_plot) == 1:
                    heights_list = [n_list]
                else:
                    heights_list = n_list

                for i, heights in enumerate(heights_list):
                    if len(data_to_plot[i]) > 0:
                        color = group_colors[i]
                        ax_hist.plot(tick_positions, heights, marker='o', linestyle='-', color=color, linewidth=2,
                                     label=f"Trend {labels[i].split(' (')[0]}")

            # --- AJOUT DES LIGNES VERTICALES POUR LES MOYENNES ET LEURS VALEURS ---
            for i, notes in enumerate(data_to_plot):
                if len(notes) > 0:
                    mean_val = notes.mean()
                    color = group_colors[i]
                    group_name = labels[i].split(' (')[0]

                    # Ligne verticale en pointillés pour la moyenne
                    ax_hist.axvline(
                        x=mean_val,
                        color=color,
                        linestyle='--',
                        linewidth=2,
                        alpha=0.9,
                        label=f"Moy. {group_name} = {mean_val:.2f}"
                    )

                    # Annotation textuelle de la valeur de la moyenne sur le graphique
                    y_max = ax_hist.get_ylim()[1]
                    # Décalage vertical étagé pour éviter la superposition des textes si les moyennes sont proches
                    y_pos = y_max * (0.88 - i * 0.08)
                    ax_hist.text(
                        mean_val, y_pos,
                        f" μ={mean_val:.2f}",
                        color=color,
                        fontweight='bold',
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=color)
                    )

            # Affichage du Box-plot en dessous avec les couleurs correspondantes
            if show_boxplot and ax_box is not None:
                valid_data = [notes.values for notes in data_to_plot if len(notes) > 0]
                valid_labels = [labels[i].split(' (')[0] for i, notes in enumerate(data_to_plot) if len(notes) > 0]
                valid_colors = [group_colors[i] for i, notes in enumerate(data_to_plot) if len(notes) > 0]

                if valid_data:
                    bp = ax_box.boxplot(valid_data, vert=False, tick_labels=valid_labels, patch_artist=True, widths=0.6)
                    for patch, color in zip(bp['boxes'], valid_colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.6)
                        patch.set_edgecolor('black')
                    ax_box.set_xlabel("Notes", fontsize=10)
                    ax_box.grid(axis='x', linestyle='--', alpha=0.5)
                    ax_box.tick_params(labelsize=9)

            photo_title_str = f" Photo : {global_photo}" if global_photo != "All" else " All photos"
            ax_hist.set_title(photo_title_str, fontsize=11, fontweight='bold')
            if not show_boxplot:
                ax_hist.set_xlabel("Notes", fontsize=10)
            ax_hist.grid(axis='y', linestyle='--', alpha=0.6)
            ax_hist.legend(loc='upper right', frameon=True, fontsize=9)

            if len(all_notes_concat.unique()) > 0:
                ax_hist.set_xticks(sorted(all_notes_concat.unique()))
        else:
            ax_hist = self.fig.add_subplot(111)
            ax_hist.text(0.5, 0.5, "Aucune donnée ne correspond\naux filtres sélectionnés",
                         horizontalalignment='center', verticalalignment='center',
                         transform=ax_hist.transAxes, fontsize=12, color='red')
            ax_hist.set_title("Comparaison des histogrammes (0 résultat)", fontsize=11, fontweight='bold')

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