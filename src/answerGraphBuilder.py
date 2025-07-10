# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 15:57:25 2025

@author: paulo
"""
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import re
from typing import List, Dict, Optional
from difflib import SequenceMatcher
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate



class GraphBuilder:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_characters(self, dialoguesDF):
        for _, row in dialoguesDF.iterrows():
            self.graph.add_node(
                row['character'],
                label='CHARACTER',
                dialogues=row['dialogues']
            )

    def add_items(self, itemsDF):
        for _, row in itemsDF.iterrows():
            self.graph.add_node(
                row['item'],
                label='ITEM',
                type=row['type'],
                description=row['description']
            )

    def add_entities(self, entitiesDF_grouped):
        for _, row in entitiesDF_grouped.iterrows():
            self.graph.add_node(
                row['entity'],
                label='ENTITY',
                entity_types=row['entity_types'],
                grouped_entities=row['grouped_entities']
            )

    def add_edges(self, entityCharDF, entityItemDF, charCharDF, charItemDF, itemCharDF, itemItemDF):
        for _, row in entityCharDF.iterrows():
            self.graph.add_edge(
                row['character'], row['entity'],
                label='mentioned_by',
                source_type='CHARACTER',
                target_type='ENTITY',
                entity_type=row['entity_type'],
                context=row['context']
            )

        for _, row in entityItemDF.iterrows():
            self.graph.add_edge(
                row['item'], row['entity'],
                label='described_in',
                source_type='ITEM',
                target_type='ENTITY',
                entity_type=row['entity_type'],
                context=row['context']
            )

        for _, row in charCharDF.iterrows():
            self.graph.add_edge(
                row['character1'], row['character2'],
                label='mentioned_by',
                source_type='CHARACTER',
                target_type='CHARACTER',
                total_interactions=row['total_interactions'],
                url=row['url']
            )

        for _, row in charItemDF.iterrows():
            self.graph.add_edge(
                row['character'], row['item'],
                label='mentioned_by',
                source_type='CHARACTER',
                target_type='ITEM',
                mentions=row['mentions'],
                context=row['context']
            )

        for _, row in itemCharDF.iterrows():
            self.graph.add_edge(
                row['item'], row['character'],
                label='described_in',
                source_type='ITEM',
                target_type='CHARACTER',
                mentions=row['mentions'],
                context=row['context']
            )

        for _, row in itemItemDF.iterrows():
            self.graph.add_edge(
                row['item1'], row['item2'],
                label='described_in',
                source_type='ITEM',
                target_type='ITEM',
                similarity_score=row['similarity_score'],
                item1_type=row['item1_type'],
                item2_type=row['item2_type']
            )

    def get_graph(self):
        return self.graph

    def plot_graph(self, node_size=500, font_size=8, figsize=(12, 10)):
        G = self.graph
        G.remove_nodes_from(list(nx.isolates(G)))  # Remove nodes with no edges
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G, seed=42)

        character_nodes = [n for n, d in G.nodes(data=True) if d.get('label') == 'CHARACTER']
        item_nodes = [n for n, d in G.nodes(data=True) if d.get('label') == 'ITEM']
        entity_nodes = [n for n, d in G.nodes(data=True) if d.get('label') == 'ENTITY']

        node_colors = {
            'CHARACTER': 'skyblue',
            'ITEM': 'orange',
            'ENTITY': 'lightgreen',
        }

        nx.draw_networkx_nodes(G, pos, nodelist=character_nodes, node_color=node_colors['CHARACTER'], node_size=node_size)
        nx.draw_networkx_nodes(G, pos, nodelist=item_nodes, node_color=node_colors['ITEM'], node_size=node_size)
        nx.draw_networkx_nodes(G, pos, nodelist=entity_nodes, node_color=node_colors['ENTITY'], node_size=node_size)

        edge_styles = {
            'mentioned_by': ('solid', 'gray'),
            'described_in': ('dashed', 'black')
        }

        for edge_type, (style, color) in edge_styles.items():
            styled_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('label') == edge_type]
            nx.draw_networkx_edges(G, pos, edgelist=styled_edges, style=style, edge_color=color)

        nx.draw_networkx_labels(G, pos, font_size=font_size)

        # Monta a legenda manualmente
        legend_elements = [
            Patch(color=node_colors['CHARACTER'], label='CHARACTER'),
            Patch(color=node_colors['ITEM'], label='ITEM'),
            Patch(color=node_colors['ENTITY'], label='ENTITY'),
            Line2D([0], [0], color='gray', lw=2, linestyle='solid', label='mentioned_by'),
            Line2D([0], [0], color='black', lw=2, linestyle='dashed', label='described_in'),
        ]

        plt.legend(handles=legend_elements, loc='upper right')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def _plot_specific_graph(self, G, node_size=500, font_size=8, figsize=(12, 10)):
        """
        Internal helper to plot any given graph G with proper colors, styles, and legend.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G, seed=42)

        # Define node categories by label
        character_nodes = [n for n, d in G.nodes(data=True) if d.get('label') == 'CHARACTER']
        item_nodes = [n for n, d in G.nodes(data=True) if d.get('label') == 'ITEM']
        entity_nodes = [n for n, d in G.nodes(data=True) if d.get('label') == 'ENTITY']

        # Color mapping per node type
        node_colors = {
            'CHARACTER': 'skyblue',
            'ITEM': 'orange',
            'ENTITY': 'lightgreen',
        }

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, nodelist=character_nodes, node_color=node_colors['CHARACTER'], node_size=node_size)
        nx.draw_networkx_nodes(G, pos, nodelist=item_nodes, node_color=node_colors['ITEM'], node_size=node_size)
        nx.draw_networkx_nodes(G, pos, nodelist=entity_nodes, node_color=node_colors['ENTITY'], node_size=node_size)

        # Define edge style by label
        edge_styles = {
            'mentioned_by': ('solid', 'gray'),
            'described_in': ('dashed', 'black')
        }

        # Draw edges by type
        for edge_type, (style, color) in edge_styles.items():
            styled_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('label') == edge_type]
            nx.draw_networkx_edges(G, pos, edgelist=styled_edges, style=style, edge_color=color)

        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=font_size)

        # Manual legend
        legend_elements = [
            Patch(color=node_colors['CHARACTER'], label='CHARACTER'),
            Patch(color=node_colors['ITEM'], label='ITEM'),
            Patch(color=node_colors['ENTITY'], label='ENTITY'),
            Line2D([0], [0], color='gray', lw=2, linestyle='solid', label='mentioned_by'),
            Line2D([0], [0], color='black', lw=2, linestyle='dashed', label='described_in'),
        ]

        plt.legend(handles=legend_elements, loc='upper right')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_subgraph_connected_to(self, entities=None, items=None, characters=None, **plot_kwargs):
        """
        Plot a subgraph that includes only the specified entities, items, or characters
        and their directly connected neighbors (incoming or outgoing).
        """
        G = self.graph

        # Collect the input nodes
        target_nodes = set()
        if entities:
            target_nodes.update(entities)
        if items:
            target_nodes.update(items)
        if characters:
            target_nodes.update([char.upper() for char in characters])

        # Find all directly connected neighbors (both directions)
        #neighbors = set()
        #for node in target_nodes:
        #    if node in G:
        #        neighbors.update(G.neighbors(node))       # outgoing edges
        #        neighbors.update(G.predecessors(node))    # incoming edges

        # Keep the targets and their direct neighbors
        #sub_nodes = target_nodes.union(neighbors)

        # Create a subgraph with just those nodes
        #subG = G.subgraph(sub_nodes).copy()
        subG = G.subgraph(target_nodes).copy()
        # Remove isolated nodes (disconnected ones, just in case)
        subG.remove_nodes_from(list(nx.isolates(subG)))

        # Plot the result using internal helper
        self._plot_specific_graph(subG, **plot_kwargs)

def returnMentionDialogue(dialoguesDF: pd.DataFrame, character: str, entity: str,
                         context_window: int = 100, case_sensitive: bool = False) -> List[Dict]:
    """
    Returns dialogue excerpts where a specific character mentions a given entity/node.

    Parameters:
    -----------
    dialoguesDF : pd.DataFrame
        DataFrame containing character dialogues with columns: character, url, dialogues
    character : str
        The character whose dialogues to search
    entity : str
        The entity/node to search for in the dialogues
    context_window : int, default=100
        Number of characters to include before and after the mention for context
    case_sensitive : bool, default=False
        Whether the search should be case-sensitive

    Returns:
    --------
    List[Dict]
        List of dictionaries containing:
        - 'character': character name
        - 'url': source URL
        - 'mention': the entity that was found
        - 'excerpt': dialogue excerpt with context
        - 'position': position of the mention in the full dialogue
    """

    # Filter dialogues for the specific character
    character_dialogues = dialoguesDF[dialoguesDF['character'] == character].copy()

    if character_dialogues.empty:
        print(f"No dialogues found for character: {character}")
        return []

    mentions = []

    # Set up search pattern based on case sensitivity
    flags = 0 if case_sensitive else re.IGNORECASE

    # Create regex pattern to find whole word matches
    # This prevents partial matches (e.g., "king" in "working")
    pattern = r'\b' + re.escape(entity) + r'\b'

    for idx, row in character_dialogues.iterrows():
        dialogue_text = str(row['dialogues'])

        # Find all mentions of the entity in this dialogue
        for match in re.finditer(pattern, dialogue_text, flags):
            start_pos = match.start()
            end_pos = match.end()

            # Calculate excerpt boundaries with context window
            excerpt_start = max(0, start_pos - context_window)
            excerpt_end = min(len(dialogue_text), end_pos + context_window)

            # Extract the excerpt
            excerpt = dialogue_text[excerpt_start:excerpt_end]

            # Add ellipsis if we're not at the beginning/end
            if excerpt_start > 0:
                excerpt = "..." + excerpt
            if excerpt_end < len(dialogue_text):
                excerpt = excerpt + "..."

            mention_info = {
                'character': row['character'],
                'url': row['url'],
                'mention': entity,
                'excerpt': excerpt.strip(),
                'position': start_pos,
                'full_dialogue_length': len(dialogue_text)
            }

            mentions.append(mention_info)

    return mentions

# Helper function to display results nicely
def displayMentions(mentions: List[Dict], max_excerpt_length: int = 200):
    """
    Display mentions in a readable format.

    Parameters:
    -----------
    mentions : List[Dict]
        List of mention dictionaries from returnMentionDialogue
    max_excerpt_length : int, default=200
        Maximum length of excerpt to display (will truncate if longer)
    """

    if not mentions:
        print("No mentions found.")
        return

    print(f"Found {len(mentions)} mention(s):\n")

    for i, mention in enumerate(mentions, 1):
        excerpt = mention['excerpt']
        if len(excerpt) > max_excerpt_length:
            excerpt = excerpt[:max_excerpt_length] + "..."

        print(f"{i}. Character: {mention['character']}")
        print(f"   Entity: {mention['mention']}")
        print(f"   URL: {mention['url']}")
        print(f"   Position: {mention['position']}")
        print(f"   Excerpt: {excerpt}")
        print("-" * 50)

def insert_line_breaks(text, words_per_line):
    """
    Inserts a line break after a specified number of words in a string.

    Args:
        text (str): The input string.
        words_per_line (int): The number of words after which a line break
                              will be inserted.

    Returns:
        str: The formatted string with line breaks.
    """
    # 1. Split the input string into a list of words.
    words = text.split()

    # 2. Create a new list to store words and line breaks.
    formatted_text = []

    # 3. Iterate through the list of words using their index and value.
    for i, word in enumerate(words):
        # Append the current word to the formatted list.
        formatted_text.append(word)

        # Check if we've reached the desired number of words for a line
        # and if it's not the very last word in the string.
        if (i + 1) % words_per_line == 0 and i + 1 < len(words):
            # Append a newline character to break the line.
            formatted_text.append('\n')

    # 4. Join the list back into a single string, using a space as the separator.
    return ' '.join(formatted_text)

def find_similar_nodes(entities, items, characters, entitiesDF_grouped, itemsDF, dialoguesDF, similarity_threshold=0.6):
    """
    Find similar nodes across dataframes using fuzzy matching.

    Parameters:
    - entities, items, characters: lists of nodes to search for
    - entitiesDF, itemsDF, dialoguesDF: dataframes to search in
    - similarity_threshold: minimum similarity score (0-1)

    Returns:
    - Dictionary with matched nodes from each dataframe
    """

    # Common words to exclude from matching
    common_words = {'the', 'to', 'of', 'and', 'a', 'an', 'in', 'on', 'at', 'by', 'for', 'with', 'from'}

    def clean_text(text):
        """Clean and normalize text for comparison"""
        if pd.isna(text):
            return ""
        return str(text).lower().strip()

    def get_meaningful_words(text):
        """Extract meaningful words from text (excluding common words)"""
        words = re.findall(r'\b\w+\b', clean_text(text))
        return [w for w in words if w not in common_words and len(w) > 1]

    def calculate_similarity(node, target):
        """Calculate similarity between two strings"""
        node_clean = clean_text(node)
        target_clean = clean_text(target)

        # Direct similarity
        direct_sim = SequenceMatcher(None, node_clean, target_clean).ratio()

        # Word-based similarity
        node_words = get_meaningful_words(node)
        target_words = get_meaningful_words(target)

        if not node_words or not target_words:
            return direct_sim

        # Check if any meaningful word from node appears in target or vice versa
        word_matches = 0
        total_comparisons = 0

        for node_word in node_words:
            for target_word in target_words:
                total_comparisons += 1
                # Check exact match or substring match
                if (node_word == target_word or
                    node_word in target_word or
                    target_word in node_word or
                    SequenceMatcher(None, node_word, target_word).ratio() > 0.8):
                    word_matches += 1

        word_sim = word_matches / total_comparisons if total_comparisons > 0 else 0

        # Combined similarity (weighted average)
        return max(direct_sim, word_sim * 0.7 + direct_sim * 0.3)


    def find_matches_in_dataframe(search_nodes, df, column_name):
        """
        Find matches for search nodes in a specific dataframe column.
        Supports cells containing lists and handles a special case for 'grouped_entities'.
        """
        matches = []

        # Check if the given column exists in the dataframe
        if column_name not in df.columns:
            print(f"Warning: Column '{column_name}' not found in dataframe")
            return matches

        for node in search_nodes:
            node_matches = []

            for idx, target in df[column_name].items():
                # Skip missing values
                if pd.isna(target):
                    continue

                # Normalize target to a list
                targets = target if isinstance(target, list) else [target]

                for item in targets:
                    similarity = calculate_similarity(node, item)

                    if similarity >= similarity_threshold:
                        # If column is 'grouped_entities', return the corresponding df['entity']
                        matched_value = df.at[idx, 'entity'] if column_name == 'grouped_entities' else item

                        node_matches.append({
                            'search_node': node,
                            'matched_value': matched_value,
                            'similarity': similarity,
                            'index': idx
                        })

            # Sort by similarity descending
            node_matches.sort(key=lambda x: x['similarity'], reverse=True)
            matches.extend(node_matches)

        return matches

    # Search all node types in all dataframes
    all_search_nodes = entities + items + characters

    results = {
        'entities_matches': find_matches_in_dataframe(all_search_nodes, entitiesDF_grouped, 'grouped_entities'),
        'items_matches': find_matches_in_dataframe(all_search_nodes, itemsDF, 'item'),
        'dialogues_matches': find_matches_in_dataframe(all_search_nodes, dialoguesDF, 'character')
    }

    return results

def display_results(results):
    """Display the results in a readable format"""
    for df_name, matches in results.items():
        print(f"\n=== {df_name.upper()} ===")
        if not matches:
            print("No matches found")
            continue

        for match in matches:
            print(f"'{match['search_node']}' -> '{match['matched_value']}' (similarity: {match['similarity']:.3f})")

# Example usage:
#entities = ['New Londo', 'Darkwraiths', 'Four Kings']
#items = ['Key to the Seal', 'Crimson Set']
#characters = ['Ingward']

# Assuming you have your dataframes loaded:
#results = find_similar_nodes(entities, items, characters, entitiesDF, itemsDF, dialoguesDF)
#display_results(results)


def get_matched_values(results):
    """Extract just the matched values from results"""
    return {
        'entity_matches': [match['matched_value'] for match in results['entities_matches']],
        'item_matches': [match['matched_value'] for match in results['items_matches']],
        'character_matches': [match['matched_value'] for match in results['dialogues_matches']]
    }

#matched_values = get_matched_values(results)



class GraphQA:
    def __init__(self, builder,dialoguesDF, itemsDF, entitiesDF_grouped, entityCharDF, entityItemDF, charCharDF, charItemDF, itemCharDF, itemItemDF, model_name="gemini-1.5-flash", temperature=0.7):
        self.builder = builder
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        self.dialoguesDF = dialoguesDF
        self.itemsDF = itemsDF
        self.entitiesDF_grouped = entitiesDF_grouped
        self.entityCharDF = entityCharDF
        self.entityItemDF = entityItemDF
        self.charCharDF = charCharDF
        self.charItemDF = charItemDF
        self.itemCharDF = itemCharDF
        self.itemItemDF = itemItemDF

        template = (
            "You are analyzing a network graph with characters, items, and entities. Answer the question using the provided data.\n"
            "Use the provided schema and context to answer the question in English.\n"
            "If the question is of the type 'who is ...?' try to find the relations of the asked entity, character or item and make a synthetization of description or dialogues"

            "ANALYSIS STEPS:\n"
            " 1. Identify key terms (nodes) in the question (characters, items, entities)\n"
            " 2. Search through dialogues for relevant conversations\n"
            " 3. Check item descriptions for relevant information\n"
            " 4. Examine relationships (listed in entityCharDF, entityItemDF, charCharDF, charItemDF, itemCharDF, itemItemDF) to understand connections\n"
            " 5. Synthesize findings into a comprehensive answer but does not hesitate to describe all useful information\n"


            "RESPONSE REQUIREMENTS:\n"
            "- The answer can be reached by deductions, since the dialogues and descriptions usually do not state the deductions, they only give some clues. However, when you are not sure, describe it as a possibility\n"
            "- Use only the provided data\n"
            "- Include all relevant entities, items, and characters in your analysis (if the entity, item or character is not listed in dialoguesDF['character'], itemsDF['item'] or entitiesDF['entity'],"
            "try to find similar entities, items or characters, with some common name AND conection between other entity, item or character.\n"
            "- Entities: Named entities listed in entitiesDF_grouped['grouped_entitites'] mentioned in the context - Return entititesDF_grouped['entity']\n"
            "- Items: Physical objects, concepts, or things listed in itemsDF['item']- Return itemsDF['item'] "
            "- Characters: People or personas involved listed in dialoguesDF['character'] - Return dialoguesDF['character']  \n"
            "- Check if the characters, items and entities cited in answer are listed in the JSON output. If it is not listed, please insert it\n"
            "- "



            "Respond ONLY in this exact JSON format:\n"
            "{{\n"
            "  \"answer\": \"...\",\n"
            "  \"entities\": [\"...\"],\n"
            "  \"items\": [\"...\"],\n"
            "  \"characters\": [\"...\"]\n"
            "}}\n"
            "Question: {question}\n"
            "Context Schema: dialoguesDF, itemsDF, entitiesDF_grouped, entityCharDF, entityItemDF, charCharDF, charItemDF, itemCharDF, itemItemDF."
        )


        prompt = PromptTemplate(input_variables=["question"], template=template)
        self.chain = LLMChain(llm=self.llm, prompt=prompt)

    def ask(self, question):
        # run LLM to get JSON
        #print("Prompt sent to LLM:")
        #print(self.chain.prompt.format(question=question))
        output = self.chain.run(question=question)
        import json

        # Extrai o JSON do output, mesmo se tiver texto antes/depois
        import re
        json_text = re.search(r'\{.*\}', output, re.DOTALL)
        if not json_text:
            raise ValueError("No JSON found in LLM output")

        data = json.loads(json_text.group())
        answer = data.get("answer", "")
        ents = data.get("entities", [])
        items = data.get("items", [])
        chars = data.get("characters", [])

        # plot subgraph
        #self.builder.plot_subgraph_connected_to(
        #    entities=ents, items=items, characters=chars, figsize=(12, 8)
        #)
        return answer, ents, items, chars

def plot_graph_and_return_answers(graphQA, question):
    answer, ents, items, chars = graphQA.ask(question)
    #answerMatches = get_matched_values(find_similar_nodes(ents, items, chars, entitiesDF, itemsDF, dialoguesDF))
    answerMatches = get_matched_values(find_similar_nodes(ents, items, chars, graphQA.entitiesDF_grouped, graphQA.itemsDF,graphQA.dialoguesDF))
    graphQA.builder.plot_subgraph_connected_to(answerMatches['entity_matches'],answerMatches['item_matches'],answerMatches['character_matches'], figsize=(12, 8))
    print(insert_line_breaks(answer,20))
    """Extract just the matched values from results"""
    return answer

def return_answers_to_neo4j(graphQA, question):
    answer, ents, items, chars = graphQA.ask(question)
    #answerMatches = get_matched_values(find_similar_nodes(ents, items, chars, entitiesDF, itemsDF, dialoguesDF))
    answerMatches = get_matched_values(find_similar_nodes(ents, items, chars, graphQA.entitiesDF_grouped, graphQA.itemsDF,graphQA.dialoguesDF))
    #graphQA.builder.plot_subgraph_connected_to(answerMatches['entity_matches'],answerMatches['item_matches'],answerMatches['character_matches'], figsize=(12, 8))
    #print(insert_line_breaks(answer,20))
    """Extract just the matched values from results"""
    return answer, answerMatches



