import os
from pathlib import Path
from typing import List, Tuple, Union, Dict
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import pdfquery
import pandas as pd
import xml.etree.ElementTree as ET
from math import sqrt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

PATH_DATA = Path(os.getcwd()) / "data" / "2_-_PDF_print_interieur/"
PATH_XML = Path(os.getcwd()) / "xml" / "2_-_PDF_print_interieur/"
PATH_IMG = Path(os.getcwd()) / "img" / "2_-_PDF_print_interieur/"
PATH_CSV = Path(os.getcwd()) / "csv" / "2_-_PDF_print_interieur/"

edgecolors = {"LTImage": "orange", "LTRect": "red", "LTCurve": "fuchsia", "LTTextBoxHorizontal": "blue",
              "LTTextLineHorizontal": "green", "LTPage": "black", "LTLine": "gray", "LTFigure": "brown",
              "LTTextLineVertical": "green", "LTTextBoxVertical": "blue"}
facecolors = {"LTImage": "gold", "LTRect": "coral", "LTCurve": "pink", "LTTextBoxHorizontal": "lightblue",
              "LTTextLineHorizontal": "lightgreen", "LTPage": "white", "LTLine": "lightgray", "LTFigure": "beige",
              "LTTextLineVertical": "lightgreen", "LTTextBoxVertical": "lightblue"}


def pdf2xml(filename: str) -> None:
    pdf = pdfquery.PDFQuery(PATH_DATA / filename)
    pdf.load()
    pdf.tree.write(PATH_XML / filename[:-4], pretty_print=True)


def build_content_dataframe(filename: str) -> None:
    tree = ET.parse(PATH_XML / filename)
    root = tree.getroot()
    for np, page in enumerate(root.findall('LTPage')):
        for i in range(5):
            pile: List[Tuple[ET, int, Union[str, None]]] = [(page, 0, np)]
            while pile:
                root, depth, id = pile[-1]
                pile = pile[:-1]
                if len(root) > 0:
                    dead_end = [item for item in root if (item.tag in ["LTRect", "LTCurve", "LTLine"] or
                            not (item.tag in ["LTImage", "LTFigure"] or item.text)) and len(item) == 0]
                    for item in dead_end:
                        root.remove(item)
                    if all([item.tag == "LTTextLineHorizontal" for item in root]):
                        root.text = "\n".join([item.text for item in root if item.text])
                        for item in root[:]:
                            root.remove(item)
                    for ix, child in enumerate(root[::-1]):
                        pile.append((child, depth+1, f"{id}-{ix}"))

        all_items = pd.DataFrame(columns=["page", "depth", "xmin", "ymin", "xmax", "ymax", "type", "text", "id"])
        page_bbox: List[float] = [float(coord) / 100 for coord in page.attrib["bbox"][1:-1].split(", ")]
        fig, ax = plt.subplots(1, figsize=(page_bbox[2], page_bbox[3]))
        plt.xlim(0, page_bbox[2])
        plt.ylim(0, page_bbox[3])
        pile: List[Tuple[ET, int, Union[str, None]]] = [(page, 0, np)]
        while pile:
            root, depth, id = pile[-1]
            ax.add_patch(paint_box(root, depth))
            dic_item: Dict[str, Union[int, float, str]] = {"depth": depth, "page": np, "xmin": root.attrib["x0"],
                                          "ymin": root.attrib["y0"], "xmax": root.attrib["x1"], "id": id,
                                          "ymax": root.attrib["y1"], "type": root.tag, "text": root.text}
            pile = pile[:-1]
            for ix, child in enumerate(root[::-1]):
                pile.append((child, depth+1, f"{id}-{ix}"))
            all_items = all_items.append(dic_item, ignore_index=True)

        plt.savefig(PATH_IMG / (filename[:-4] + f"_{np}.png"))
        plt.close()
        all_items.to_csv(PATH_CSV / (filename[:-4] + f"_{np}.csv"), sep=";", encoding="utf-8")


def paint_box(item: ET.Element, depth: int):
    bbox: List[float] = [float(coord) / 100 for coord in item.attrib["bbox"][1:-1].split(", ")]
    rectangle = Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                          edgecolor=edgecolors[item.tag], facecolor=facecolors[item.tag], alpha=0.5,
                          linestyle="dashed" if depth == 2 else "dotted" if depth >= 3 else "solid")
    return rectangle


def build_relation_table():
    all_dataframes: List[pd.DataFrame] = sorted([pd.read_csv(PATH_CSV / adr, encoding="utf-8", sep=";", index_col=0)
                                          for adr in os.listdir(PATH_CSV)], key=lambda z: z.page.unique()[0])
    for page in all_dataframes[:1]:
        page = page.drop(["page", "text"], axis=1).drop(0, axis=0)
        distances: pd.DataFrame = page.merge(page, how="cross", suffixes=("_1", "_2"))
        distances = distances.loc[distances.id_1 != distances.id_2, :]
        distances.loc[:, "siblings"] = distances.apply(lambda z: len(z["id_1"].split("-")) > 2 and
                                       (z["id_1"].split("-")[:-1] == z["id_2"].split("-")[:-1]), axis=1)
        distances.loc[:, "is_child"] = distances.apply(lambda z: z["id_1"].split("-")[:-1] == z["id_2"].split("-"),
                                                       axis=1)
        distances.loc[:, "is_parent"] = distances.apply(lambda z: z["id_2"].split("-")[:-1] == z["id_1"].split("-"),
                                                        axis=1)
        distances.loc[:, "distance_geom"] = distances.apply(distances_btw_boxes, axis=1)
        distances.to_csv("distances.csv", encoding="utf-8", sep=";", index=False)


def distances_btw_boxes(boxes: pd.Series) -> float:
    distx: List[float] = [boxes["xmin_1"]-boxes["xmin_2"], boxes["xmax_1"]-boxes["xmin_2"],
                          boxes["xmin_1"]-boxes["xmax_2"], boxes["xmax_1"]-boxes["xmax_2"]]
    disty: List[float] = [boxes["ymin_1"]-boxes["ymin_2"], boxes["ymax_1"]-boxes["ymin_2"],
                          boxes["ymin_1"]-boxes["ymax_2"], boxes["ymax_1"]-boxes["ymax_2"]]
    inter_x = len([d for d in distx if d < 0])
    inter_y = len([d for d in disty if d < 0])
    min_x = min([abs(d) for d in distx])
    min_y = min([abs(d) for d in disty])
    if 1 <= inter_x <= 3 and 1 <= inter_y <= 3:
        return 0
    elif 1 <= inter_x <= 3:
        return min_y
    elif 1 <= inter_y <= 3:
        return min_x
    else:
        return sqrt(min_x**2 + min_y**2)


def predict_siblings():
    distances: pd.DataFrame = pd.read_csv("distances.csv", encoding="utf-8", sep=";")
    Y = distances.siblings
    oec = OneHotEncoder(sparse=False)
    X0 = oec.fit_transform(distances.loc[:, ["type_1", "type_2"]])
    X = pd.DataFrame(X0).merge(distances.distance_geom.to_frame(), left_index=True, right_index=True)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=(list(oec.get_feature_names_out()) + ["dist_geom"]))
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)

    rfc = RandomForestClassifier(class_weight="balanced")
    rfc.fit(Xtrain, Ytrain)
    Ypred = rfc.predict(Xtest)
    print(rfc.score(Xtest, Ytest))
    print(recall_score(Ypred, Ytest))
    print(precision_score(Ypred, Ytest))

    distances.loc[:, "siblings_pred"] = rfc.predict(X)
    distances.to_csv("distances_with_pred.csv", encoding="utf-8", sep=";", index=False)


def draw_predictions():
    distances: pd.DataFrame = pd.read_csv("distances_with_pred.csv", encoding="utf-8", sep=";")
    fig, ax = plt.subplots(1, figsize=(10, 17))
    plt.xlim(0, 600)
    plt.ylim(0, 850)
    boxes = distances.drop_duplicates(subset=["id_1"])
    for i, row in boxes.iterrows():
        rectangle = Rectangle((row["xmin_1"], row["ymin_1"]), row["xmax_1"] - row["xmin_1"],
                              row["ymax_1"] - row["ymin_1"],
                              edgecolor=edgecolors[row["type_1"]], facecolor=facecolors[row["type_1"]], alpha=0.5)
        ax.add_patch(rectangle)
    for i, row in distances.iterrows():
        if row['siblings']:
            plt.plot([(row["xmin_1"]+row["xmax_1"])/2, (row["xmin_2"]+row["xmax_2"])/2],
                              [(row["ymin_1"]+row["ymax_1"])/2, (row["ymin_2"]+row["ymax_2"])/2],
                     linewidth=1, color="gray")
    plt.savefig("ground_truth.png")
    plt.close()

    fig, ax = plt.subplots(1, figsize=(10, 17))
    plt.xlim(0, 600)
    plt.ylim(0, 850)
    boxes = distances.drop_duplicates(subset=["id_1"])
    for i, row in boxes.iterrows():
        rectangle = Rectangle((row["xmin_1"], row["ymin_1"]), row["xmax_1"] - row["xmin_1"],
                              row["ymax_1"] - row["ymin_1"],
                              edgecolor=edgecolors[row["type_1"]], facecolor=facecolors[row["type_1"]], alpha=0.5)
        ax.add_patch(rectangle)
    for i, row in distances.iterrows():
        if row['siblings_pred']:
            plt.plot([(row["xmin_1"]+row["xmax_1"])/2, (row["xmin_2"]+row["xmax_2"])/2],
                              [(row["ymin_1"]+row["ymax_1"])/2, (row["ymin_2"]+row["ymax_2"])/2],
                     linewidth=1, color="gray")
    plt.savefig("prediction.png")
    plt.close()


if __name__ == '__main__':
    # pdf2xml("09172531_032-065_Chap1.pdf")
    # build_content_dataframe("09172531_032-065_Chap1.xml")
    # build_relation_table()
    # predict_siblings()
    draw_predictions()
