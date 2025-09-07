import graphviz
import os
import sys
import shutil

# Ensure the output directory exists
os.makedirs("architecture_diagrams", exist_ok=True)

def set_pretty_styles(rankdir='LR'):
    """Defines a more aesthetically pleasing style for the diagrams."""
    return {
        'graph_attr': {
            'rankdir': rankdir,       # Layout direction (LR or TB)
            'splines': 'ortho',
            'nodesep': '0.8',
            'ranksep': '1.5',
            'fontname': 'Helvetica, Arial, sans-serif',
            'fontsize': '26',
            'fontcolor': '#2D3748',
            'bgcolor': '#F7FAFC',
        },
        'node_attr': {
            'shape': 'box',
            'style': 'rounded,filled',
            'fontname': 'Helvetica, Arial, sans-serif',
            'fontsize': '18',
            'color': '#CBD5E0',
            'fontcolor': '#1A202C',
            'margin': '0.4,0.3',
            'shadow': 'true',
        },
        'edge_attr': {
            'fontname': 'Helvetica, Arial, sans-serif',
            'fontsize': '14',
            'color': '#718096',
            'arrowsize': '1.0',
            'penwidth': '2.0'
        }
    }

def generate_baseline_diagram():
    """Generates the architecture diagram for the Baseline CNN model."""
    styles = set_pretty_styles(rankdir='LR')
    dot = graphviz.Digraph(
        'Baseline_Architecture',
        graph_attr=styles['graph_attr'],
        node_attr=styles['node_attr'],
        edge_attr=styles['edge_attr']
    )
    dot.attr(label='Architecture 1: Baseline Model (Horizontal)')
    dot.node('input', 'MNIST Image\n(28x28)', fillcolor='#EBF8FF:#BEE3F8', gradientangle='0')
    with dot.subgraph(name='cluster_feature_extractor') as c:
        c.attr(label='Feature Extractor (SingleDigitClassifier)', style='rounded', color='#A0AEC0', bgcolor='#EDF2F7', fontcolor='#2D3748', fontsize='22')
        c.node('conv1', 'Conv2D → ReLU → MaxPool', fillcolor='#FEEBC8:#FBD38D', gradientangle='0')
        c.node('conv2', 'Conv2D → ReLU → MaxPool', fillcolor='#FEEBC8:#FBD38D', gradientangle='0')
        c.node('fc1', 'Linear → Tanh\n(100 Features)', fillcolor='#FBD38D:#F6AD55', gradientangle='0')
        c.edges([('conv1', 'conv2'), ('conv2', 'fc1')])
    dot.node('classifier', 'Classifier Head\n(Linear → 10 Logits)', fillcolor='#E9D8FD:#D6BCFA', gradientangle='0')
    dot.node('output', 'Predicted Digit\n(0-9)', fillcolor='#C6F6D5:#9AE6B4', gradientangle='0')
    dot.edge('input', 'conv1')
    dot.edge('fc1', 'classifier')
    dot.edge('classifier', 'output')
    output_path = os.path.join("architecture_diagrams", "1_baseline_architecture_horizontal_pretty")
    dot.render(output_path, format='png', cleanup=True)
    print(f"Prettier Horizontal Baseline diagram saved to {output_path}.png")

def generate_ltn_diagram(name, logic_type, operator_details):
    """Generates a generalized diagram for an LTN model."""
    styles = set_pretty_styles(rankdir='TB')
    dot = graphviz.Digraph(f'{name.replace(" ", "_")}_Architecture', graph_attr=styles['graph_attr'], node_attr=styles['node_attr'], edge_attr=styles['edge_attr'])
    dot.attr(label=f'Architecture: {name}')
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(style='invis')
        c.node('input1', 'Image A', fillcolor='#EBF8FF:#BEE3F8', gradientangle='270')
        c.node('input2', 'Image B', fillcolor='#EBF8FF:#BEE3F8', gradientangle='270')
    with dot.subgraph(name='cluster_cnn') as c:
        c.attr(label='Shared CNN Feature Extractor', style='rounded', color='#A0AEC0', bgcolor='#EDF2F7', fontcolor='#2D3748', fontsize='20')
        c.node('cnn1', 'Feature Extractor A', fillcolor='#FBD38D:#F6AD55', gradientangle='270')
        c.node('cnn2', 'Feature Extractor B', fillcolor='#FBD38D:#F6AD55', gradientangle='270')
    with dot.subgraph(name='cluster_ltn') as c:
        c.attr(label='Logic Tensor Network Framework', style='rounded', color='#805AD5', bgcolor='#FAF5FF', fontcolor='#5A67D8', fontsize='22')
        c.node('grounding', 'Grounding\n(Features → Logic)', fillcolor='#E9D8FD:#D6BCFA', gradientangle='270')
        c.node('logic', f'''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="4" BGCOLOR="#F7FAFC"><TR><TD><B>Axiom Satisfaction</B></TD></TR><TR><TD ALIGN="LEFT" BALIGN="LEFT" BGCOLOR="#EDF2F7"><FONT FACE="Menlo, Monaco, 'Courier New', monospace" POINT-SIZE="14" COLOR="#4A5568">∀x,y,z ∃d₁,d₂ : (model(x, d₁) ∧ model(y, d₂))<BR/>where (d₁.value + d₂.value = z.value)</FONT></TD></TR><TR><TD><B>{logic_type}</B></TD></TR><TR><TD ALIGN="LEFT" BALIGN="LEFT" BGCOLOR="#EDF2F7">{operator_details}</TD></TR></TABLE>>''', shape='plaintext')
        c.node('loss', 'Loss = 1 - Sat(Axiom)', fillcolor='#E9D8FD:#D6BCFA', gradientangle='270')
    dot.edge('input1', 'cnn1')
    dot.edge('input2', 'cnn2')
    dot.edge('cnn1', 'grounding')
    dot.edge('cnn2', 'grounding')
    dot.edge('grounding', 'logic')
    dot.edge('logic', 'loss')
    output_path = os.path.join("architecture_diagrams", f'{name.replace(" ", "_").replace(".", "").lower()}_architecture_pretty')
    dot.render(output_path, format='png', cleanup=True)
    print(f"Prettier {name} diagram saved to {output_path}.png")

def generate_probing_diagram():
    """Generates a diagram explaining Linear Probing."""
    styles = set_pretty_styles(rankdir='TB')
    dot = graphviz.Digraph('Probing_Diagram', graph_attr=styles['graph_attr'], node_attr=styles['node_attr'], edge_attr=styles['edge_attr'])
    dot.attr(label='Analysis 1: Linear Probing')
    
    with dot.subgraph(name='cluster_main_model') as c:
        c.attr(label='Pre-trained Model', style='rounded', color='#A0AEC0', bgcolor='#EDF2F7', fontcolor='#2D3748', fontsize='22')
        c.node('input', 'Image', fillcolor='#EBF8FF:#BEE3F8', gradientangle='270')
        c.node('conv_layers', 'Conv Layers', fillcolor='#FEEBC8:#FBD38D', gradientangle='270')
        c.node('features', 'Feature Vector\n(Layer is "frozen")', fillcolor='#FBD38D:#F6AD55', gradientangle='270')
        c.node('original_output', 'Original Output', fillcolor='#C6F6D5:#9AE6B4', gradientangle='270')
        c.edges([('input', 'conv_layers'), ('conv_layers', 'features'), ('features', 'original_output')])

    with dot.subgraph(name='cluster_probe') as c:
        c.attr(label='Probe', style='rounded', color='#DD6B20', bgcolor='#FFF5EB', fontcolor='#C05621', fontsize='22')
        c.node('probe_classifier', 'Simple Linear Classifier\n(Trainable)', fillcolor='#F6AD55:#ED8936', gradientangle='270')
        c.node('probe_accuracy', 'Probe Accuracy (%)', fillcolor='#F6AD55:#ED8936', gradientangle='270')
        c.edge('probe_classifier', 'probe_accuracy')

    dot.edge('features', 'probe_classifier', style='dashed', penwidth='3.0', color='#DD6B20', label=' "How well do features\nencode digit identity?"')
    
    output_path = os.path.join("architecture_diagrams", "4_linear_probing_diagram")
    dot.render(output_path, format='png', cleanup=True)
    print(f"Linear Probing diagram saved to {output_path}.png")
    
def generate_vae_diagram():
    """Generates a diagram explaining VAE Latent Space analysis."""
    styles = set_pretty_styles(rankdir='TB')
    dot = graphviz.Digraph('VAE_Diagram', graph_attr=styles['graph_attr'], node_attr=styles['node_attr'], edge_attr=styles['edge_attr'])
    dot.attr(label='Analysis 2: VAE Latent Space (Vertical)')
    
    dot.node('features', 'Feature Vector\n(from CNN)', fillcolor='#FBD38D:#F6AD55', gradientangle='270')
    
    with dot.subgraph(name='cluster_vae') as c:
        c.attr(label='VAE Probe', style='rounded', color='#38A169', bgcolor='#F0FFF4', fontcolor='#2F855A', fontsize='22')
        c.node('encoder', 'Encoder', fillcolor='#9AE6B4:#68D391', gradientangle='270')
        c.node('latent', 'Latent Space\n(z₁, z₂)', shape='ellipse', fillcolor='#C6F6D5:#9AE6B4', gradientangle='270')
        c.node('decoder', 'Decoder', fillcolor='#9AE6B4:#68D391', gradientangle='270')

    dot.node('reconstructed', 'Reconstructed\nFeature Vector', fillcolor='#FBD38D:#F6AD55', gradientangle='270')

    dot.edge('features', 'encoder', label=' "Create an organized\nmap of digits" ')
    dot.edge('encoder', 'latent')
    dot.edge('latent', 'decoder')
    dot.edge('decoder', 'reconstructed')

    output_path = os.path.join("architecture_diagrams", "5_vae_diagram_vertical")
    dot.render(output_path, format='png', cleanup=True)
    print(f"VAE Latent Space (Vertical) diagram saved to {output_path}.png")

def generate_causal_diagram():
    """Generates a diagram for Causal Intervention (Activation Patching)."""
    styles = set_pretty_styles(rankdir='TB')
    dot = graphviz.Digraph('Causal_Diagram', graph_attr=styles['graph_attr'], node_attr=styles['node_attr'], edge_attr=styles['edge_attr'])
    dot.attr(label='Analysis 3: Causal Intervention (Activation Patching)')
    
    with dot.subgraph(name='cluster_models') as c:
        c.attr(style='invis')
        # Bad Model
        with c.subgraph(name='cluster_bad') as bad:
            bad.attr(label='Bad Model (Baseline)', style='rounded', color='#E53E3E', bgcolor='#FFF5F5', fontcolor='#C53030', fontsize='20')
            bad.node('input_bad', 'Input Image ("3")', fillcolor='#FED7D7:#FC8181', gradientangle='270')
            bad.node('cnn_bad', 'CNN', fillcolor='#FC8181:#F56565', gradientangle='270')
            bad.node('fc1_bad', 'fc1 Activation', fillcolor='#FEB2B2', gradientangle='270')
            bad.node('output_bad', 'Output: "8"\n(Incorrect)', fillcolor='#C53030', fontcolor='white')
            bad.edges([('input_bad', 'cnn_bad'), ('cnn_bad', 'fc1_bad'), ('fc1_bad', 'output_bad')])
        
        # Good Model
        with c.subgraph(name='cluster_good') as good:
            good.attr(label='Good Model (Stable LTN)', style='rounded', color='#38A169', bgcolor='#F0FFF4', fontcolor='#2F855A', fontsize='20')
            good.node('input_good', 'Input Image ("3")', fillcolor='#C6F6D5:#9AE6B4', gradientangle='270')
            good.node('cnn_good', 'CNN', fillcolor='#9AE6B4:#68D391', gradientangle='270')
            good.node('fc1_good', 'fc1 Activation', fillcolor='#B2F5EA', gradientangle='270')
            good.node('output_good', 'Output: "3"\n(Correct)', fillcolor='#2F855A', fontcolor='white')
            good.edges([('input_good', 'cnn_good'), ('cnn_good', 'fc1_good'), ('fc1_good', 'output_good')])

    # Patched Model
    with dot.subgraph(name='cluster_patched') as patched:
        patched.attr(label='Patched Bad Model', style='rounded', color='#4299E1', bgcolor='#EBF8FF', fontcolor='#2B6CB0', fontsize='20')
        patched.node('output_patched', 'Output: "3"\n(Corrected!)', fillcolor='#63B3ED', fontcolor='white')

    # Causal link
    dot.edge('fc1_good', 'fc1_bad', style='dashed', penwidth='3.5', color='#DD6B20', dir='back',
             label=' "Is this feature causally responsible?"\nPatch Activation ')
    dot.edge('fc1_bad', 'output_patched', style='dashed', penwidth='2.5', color='#4299E1')

    output_path = os.path.join("architecture_diagrams", "6_causal_intervention_diagram")
    dot.render(output_path, format='png', cleanup=True)
    print(f"Causal Intervention diagram saved to {output_path}.png")


if __name__ == '__main__':
    if not shutil.which('dot'):
        print("--- ERROR ---", file=sys.stderr)
        print("Graphviz executable not found. The 'dot' command is required to generate diagrams.", file=sys.stderr)
        print("\nPlease install the Graphviz software package to resolve this.", file=sys.stderr)
        print("  - For Conda environments, try: conda install -c anaconda python-graphviz", file=sys.stderr)
        print("  - For Debian/Ubuntu, try: sudo apt-get install graphviz", file=sys.stderr)
        sys.exit(1)

    print("--- Generating Architecture Diagrams ---")
    generate_baseline_diagram()
    fuzzy_operators = '''<FONT FACE="Menlo, Monaco, 'Courier New', monospace" POINT-SIZE="14" COLOR="#4A5568"><B>AND (∧):</B> Lukasiewicz T-Norm<BR/><I>max(0, a + b - 1)</I><BR/><BR/><B>Quantifiers (∀, ∃):</B><BR/><I>AggregMean / AggregMeanError</I></FONT>'''
    generate_ltn_diagram(name="2. Fuzzy LTN", logic_type="Lukasiewicz Logic", operator_details=fuzzy_operators)
    stable_operators = '''<FONT FACE="Menlo, Monaco, 'Courier New', monospace" POINT-SIZE="14" COLOR="#4A5568"><B>AND (∧):</B> Product T-Norm<BR/><I>π(π(x) * π(y))</I><BR/><BR/><B>Quantifiers (∀, ∃):</B><BR/><I>AggregPMean / AggregPMeanError</I></FONT>'''
    generate_ltn_diagram(name="3. Stable LTN", logic_type="Stable Logic", operator_details=stable_operators)
    
    print("\n--- Generating Analysis Technique Diagrams ---")
    generate_probing_diagram()
    generate_vae_diagram()
    generate_causal_diagram()

    print("\n--- All diagrams generated successfully in the 'architecture_diagrams' folder. ---")
