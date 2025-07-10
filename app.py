import streamlit as st
from pyvis.network import Network
from neo4j import GraphDatabase
import streamlit.components.v1 as components
import os
import pandas as pd
import time
import sys

sys.path.append('./src/')
from answerGraphBuilder import GraphBuilder, GraphQA, return_answers_to_neo4j

# Load CSV data
adressDialoguesCsv = 'data/darksouls_characters_dialogues.csv'
adressItemsCsv = 'data/darksouls_item_descriptions.csv'
adressEntitiesCsv = 'data/entities_df.csv'
adressEntitiesGroupedCsv = 'data/entities_grouped_df.csv'
adressEntityCharCsv = 'data/entity_char_df.csv'
adressEntityItemCsv = 'data/entity_item_df.csv'
adressCharCharCsv = 'data/char_char_df.csv'
adressCharItemCsv = 'data/char_item_df.csv'
adressItemCharCsv = 'data/item_char_df.csv'
adressItemItemCsv = 'data/item_item_df.csv'

dialoguesDF = pd.read_csv(adressDialoguesCsv)
itemsDF = pd.read_csv(adressItemsCsv)
entitiesDF = pd.read_csv(adressEntitiesCsv)
entitiesDF_grouped = pd.read_csv(adressEntitiesGroupedCsv)
entityCharDF = pd.read_csv(adressEntityCharCsv)
entityItemDF = pd.read_csv(adressEntityItemCsv)
charCharDF = pd.read_csv(adressCharCharCsv)
charItemDF = pd.read_csv(adressCharItemCsv)
itemCharDF = pd.read_csv(adressItemCharCsv)
itemItemDF = pd.read_csv(adressItemItemCsv)

dialoguesDF = dialoguesDF.rename(columns={'name': 'character'})
itemsDF = itemsDF.rename(columns={'name': 'item'})

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
NEO4JPASSWORD = st.secrets["NEO4JPASSWORD"]


# Neo4j config
uri = "neo4j+s://5aef28dd.databases.neo4j.io"
user = "neo4j"
# .env
password = NEO4JPASSWORD
driver = GraphDatabase.driver(uri, auth=(user, password))


# Set Gemini API KEY
# .env
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# UI setup
st.set_page_config(layout="wide")
st.title("Neo4j Graph QA Browser")

# Form input
with st.form(key="query_form"):
    option = st.selectbox("Choose input mode:", ["LLM", "cypher"])
    prompt = st.text_area("Enter your question or Cypher query:", height=200)
    submitted = st.form_submit_button("Submit")

builder = GraphBuilder()
builder.add_characters(dialoguesDF)
builder.add_items(itemsDF)
builder.add_entities(entitiesDF_grouped)
builder.add_edges(entityCharDF, entityItemDF, charCharDF, charItemDF, itemCharDF, itemItemDF)

qa = GraphQA(builder, dialoguesDF, itemsDF, entitiesDF_grouped, entityCharDF, entityItemDF, charCharDF, charItemDF, itemCharDF, itemItemDF, model_name="models/gemini-2.5-flash", temperature=0)

def get_node_color(labels):
    if "CHARACTER" in labels:
        return "skyblue"
    elif "ITEM" in labels:
        return "orange"
    elif "ENTITY" in labels:
        return "lightgreen"
    else:
        return "gray"

def get_edge_style(relation):
    if relation == "mentioned_by":
        return {"color": "gray", "dashes": False}
    elif relation == "described_in":
        return {"color": "black", "dashes": True}
    else:
        return {"color": "lightgray", "dashes": False}

def build_graph(results):
    g = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    g.barnes_hut(gravity=-25000, central_gravity=0.3, spring_length=200, damping=0.85)
    g.set_options("""
var options = {
  "nodes": {
    "shape": "dot",
    "size": 25,
    "font": {"size": 14}
  },
  "edges": {
    "smooth": true,
    "arrows": {
      "to": {"enabled": true, "scaleFactor": 0.5}
    }
  },
  "physics": {
    "enabled": true,
    "barnesHut": {
      "gravitationalConstant": -25000,
      "springLength": 200,
      "springConstant": 0.03,
      "damping": 0.85
    }
  }
}
""")

    added_nodes = set()
    for record in results:
        source = record.get("source")
        target = record.get("target")
        relation = record.get("relation")
        if not source or not target or not relation:
            continue

        source_color = get_node_color(record.get("source_labels", []))
        target_color = get_node_color(record.get("target_labels", []))
        edge_style = get_edge_style(relation)

        if source not in added_nodes:
            g.add_node(source, label=source, color=source_color)
            added_nodes.add(source)
        if target not in added_nodes:
            g.add_node(target, label=target, color=target_color)
            added_nodes.add(target)

        g.add_edge(source, target, color=edge_style["color"], dashes=edge_style["dashes"], title=relation)

    return g

def run_cypher_query(tx, cypher):
    result = tx.run(cypher)
    return [record.data() for record in result]

# Create layout columns
col1, col2 = st.columns([4, 1])

with col2:
    st.markdown("""
    ### Legend
    **Node Colors**  
    🟦 CHARACTER = skyblue  
    🟧 ITEM = orange  
    🟩 ENTITY = lightgreen  

    **Edge Styles**  
    ━━━ `mentioned_by` = solid gray  
    ━ ━ `described_in` = dashed black  
    """)

with col1:
    graph_data = []
    show_graph = False

    if submitted and prompt:
        try:
            if option == "LLM":
                start_time = time.time()
                answer, nodesDict = return_answers_to_neo4j(qa, prompt)
                duration = time.time() - start_time

                if duration > 60:
                    st.warning("Can't answer this question now.")
                else:
                    nodeList = list(set(sum(nodesDict.values(), [])))
                    st.subheader("Interpretative Answer:")
                    st.markdown(answer)
                    with driver.session() as session:
                        graph_data = session.read_transaction(
                            lambda tx: run_cypher_query(tx, f"""
                                MATCH (n)-[r]-(m)
                                WHERE n.name IN [{', '.join(f'"{n}"' for n in nodeList)}]
                                  AND m.name IN [{', '.join(f'"{n}"' for n in nodeList)}]
                                RETURN DISTINCT n.name AS source, type(r) AS relation, m.name AS target,
                                               labels(n) AS source_labels, labels(m) AS target_labels
                            """)
                        )
                        show_graph = True

            elif option == "cypher":
                with driver.session() as session:
                    result = session.run(prompt)
                    graph_data = []
                    all_nodes = set()
                    edges = []

                    for record in result:
                        for value in record.values():
                            if hasattr(value, "nodes") and hasattr(value, "relationships"):
                                all_nodes.update(value.nodes)
                                edges.extend(value.relationships)
                            elif hasattr(value, "start_node") and hasattr(value, "end_node") and hasattr(value, "type"):
                                edges.append(value)
                                all_nodes.add(value.start_node)
                                all_nodes.add(value.end_node)
                            elif hasattr(value, "labels"):
                                all_nodes.add(value)

                    used_nodes = set()

                    for rel in edges:
                        start = rel.start_node
                        end = rel.end_node
                        graph_data.append({
                            "source": start.get("name", str(start.id)),
                            "target": end.get("name", str(end.id)),
                            "relation": rel.type,
                            "source_labels": list(start.labels),
                            "target_labels": list(end.labels)
                        })
                        used_nodes.add(start)
                        used_nodes.add(end)

                    for node in all_nodes - used_nodes:
                        graph_data.append({
                            "source": node.get("name", str(node.id)),
                            "target": None,
                            "relation": None,
                            "source_labels": list(node.labels),
                            "target_labels": []
                        })

                    if graph_data:
                        show_graph = True
                    else:
                        st.warning("No nodes or relationships found to display.")

        except Exception as e:
            st.error("Error processing request.")

    elif not prompt:
        with driver.session() as session:
            graph_data = session.read_transaction(lambda tx: run_cypher_query(tx, """
                MATCH (n)-[r]-(m)
                RETURN DISTINCT n.name AS source, type(r) AS relation, m.name AS target,
                                labels(n) AS source_labels, labels(m) AS target_labels
                LIMIT 200
            """))
            show_graph = True

    if show_graph:
        net = build_graph(graph_data)
        net.save_graph("graph.html")
        HtmlFile = open("graph.html", "r", encoding="utf-8")
        source_code = HtmlFile.read()
        components.html(source_code, height=650, width=1000)
