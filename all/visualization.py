import pandas as pd
import networkx as nx
import math
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from config_manager import ConfigManager

config_manager = ConfigManager()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def build_network(excel_path: str):
    """构建语义网络，返回节点列表和边列表"""
    df = pd.read_excel(excel_path)
    # 期望列：原始文本、实体标签、转写结果
    texts = df.iloc[:, 0].astype(str).tolist()
    labels = df.iloc[:, 1].astype(str).tolist() if df.shape[1] > 1 else [""]*len(df)
    rewrites = df.iloc[:, 2].astype(str).tolist() if df.shape[1] > 2 else [""]*len(df)

    nodes = []
    for i, (txt, lbl, rw) in enumerate(zip(texts, labels, rewrites)):
        nodes.append({
            "id": i,
            "label": f"语段{i+1}",
            "text": txt[:50] + "..." if len(txt)>50 else txt,
            "raw_text": txt,
            "labels": lbl,
            "rewrite": rw
        })

    # 计算实体共现边
    # 提取所有实体（简单按空格和逗号分割）
    all_entities = set()
    entity_list_per_node = []
    for lbl in labels:
        if pd.isna(lbl) or lbl == "":
            entity_list_per_node.append([])
            continue
        # 分割实体，支持顿号、逗号、空格
        entities = [e.strip() for e in re.split(r"[、，,;；\s]+", lbl) if e.strip()]
        entity_list_per_node.append(entities)
        all_entities.update(entities)

    # 计算每个实体的文档频率
    df_entity = {}
    N = len(nodes)
    for ent in all_entities:
        cnt = sum(1 for ents in entity_list_per_node if ent in ents)
        df_entity[ent] = cnt

    # 构建实体共现边
    cooccur_edges = []
    for i in range(N):
        for j in range(i+1, N):
            if i == j:
                continue
            ents_i = entity_list_per_node[i]
            ents_j = entity_list_per_node[j]
            common = set(ents_i) & set(ents_j)
            if common:
                # 计算权重：共享实体数 * 逆文档频率加权
                weight = 0
                for ent in common:
                    idf = math.log(N / (df_entity[ent] + 1)) + 1
                    weight += idf
                # 归一化？暂不归一化，直接使用
                cooccur_edges.append({
                    "source": i,
                    "target": j,
                    "weight": weight,
                    "type": "cooccur"
                })

    # 计算语义相似边
    # 使用原始文本+转写结果组合作为语义表示
    texts_for_embed = []
    for i in range(N):
        comb = texts[i] + " " + rewrites[i]
        texts_for_embed.append(comb)

    vecs = embedding_model.encode(texts_for_embed, show_progress_bar=False)
    sim_matrix = cosine_similarity(vecs)

    semantic_edges = []
    threshold = 0.7
    for i in range(N):
        for j in range(i+1, N):
            sim = sim_matrix[i][j]
            if sim >= threshold:
                semantic_edges.append({
                    "source": i,
                    "target": j,
                    "weight": sim,
                    "type": "semantic"
                })

    # 合并边，标记同时存在的边为混合类型
    edge_dict = {}
    for e in cooccur_edges:
        key = (e["source"], e["target"])
        edge_dict[key] = {"cooccur": e["weight"], "semantic": 0, "mixed": False}
    for e in semantic_edges:
        key = (e["source"], e["target"])
        if key in edge_dict:
            edge_dict[key]["semantic"] = e["weight"]
            edge_dict[key]["mixed"] = True
        else:
            edge_dict[key] = {"cooccur": 0, "semantic": e["weight"], "mixed": False}

    edges = []
    for (s,t), w in edge_dict.items():
        if w["mixed"]:
            etype = "mixed"
            weight = (w["cooccur"] + w["semantic"]) / 2  # 取平均作为混合权重
        elif w["cooccur"] > 0:
            etype = "cooccur"
            weight = w["cooccur"]
        else:
            etype = "semantic"
            weight = w["semantic"]
        edges.append({
            "source": s,
            "target": t,
            "weight": weight,
            "type": etype
        })

    return nodes, edges

def export_gephi(nodes, edges, filename="network.gexf"):
    """导出Gephi可读的GEXF文件"""
    import xml.etree.ElementTree as ET
    from xml.dom import minidom

    gephi_root = ET.Element("gexf", xmlns="http://www.gexf.net/1.2draft", version="1.2")
    graph = ET.SubElement(gephi_root, "graph", mode="static", defaultedgetype="undirected")

    # 节点
    attributes = ET.SubElement(graph, "attributes", {"class": "node", "mode": "static"})
    attr_text = ET.SubElement(attributes, "attribute", id="text", title="text", type="string")
    attr_labels = ET.SubElement(attributes, "attribute", id="labels", title="labels", type="string")
    attr_rewrite = ET.SubElement(attributes, "attribute", id="rewrite", title="rewrite", type="string")

    nodes_elem = ET.SubElement(graph, "nodes")
    for n in nodes:
        node = ET.SubElement(nodes_elem, "node", id=str(n["id"]), label=n["label"])
        attvalues = ET.SubElement(node, "attvalues")
        ET.SubElement(attvalues, "attvalue", for_="text", value=n["raw_text"])
        ET.SubElement(attvalues, "attvalue", for_="labels", value=n["labels"])
        ET.SubElement(attvalues, "attvalue", for_="rewrite", value=n["rewrite"])

    # 边
    edges_elem = ET.SubElement(graph, "edges")
    for i, e in enumerate(edges):
        edge = ET.SubElement(edges_elem, "edge", id=str(i), source=str(e["source"]), target=str(e["target"]), weight=str(e["weight"]))
        # 添加类型属性
        attvalues = ET.SubElement(edge, "attvalues")
        ET.SubElement(attvalues, "attvalue", for_="type", value=e["type"])

    # 美化输出
    rough_string = ET.tostring(gephi_root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")

    output_dir = config_manager.get("output_dir")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)
    return save_path