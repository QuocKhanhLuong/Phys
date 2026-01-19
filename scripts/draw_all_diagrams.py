import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

OUTPUT_DIR = 'walkthrough'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def draw_box(ax, x, y, w, h, label, color='white', edgecolor='black'):
    rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=edgecolor, facecolor=color, zorder=10)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=10, fontweight='bold', zorder=11)
    return x+w, y+h/2  # Return right-center for connection

def draw_arrow(ax, x1, y1, x2, y2, label=None, color='black'):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), 
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.1, label, ha='center', va='bottom', fontsize=9, color=color)

def draw_encoder_block():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    
    # Input
    draw_box(ax, 0.5, 3.5, 2, 1, "Input\nFeatures", '#E0E0E0')
    draw_arrow(ax, 2.5, 4, 3.5, 4) # Split
    
    # ePURE Branch
    draw_arrow(ax, 3.5, 4, 4.5, 6)
    draw_box(ax, 4.5, 5.5, 2, 1, "ePURE\n(Noise Est.)", '#FFD700')
    
    # Main path
    draw_arrow(ax, 3.5, 4, 6.5, 4)
    # Smoothing
    draw_arrow(ax, 6.5, 6, 7.5, 4.5, "noise_profile") # from ePURE
    draw_box(ax, 7, 3.5, 2, 1, "Adaptive\nSmoothing", '#90EE90')
    
    # Convs
    draw_arrow(ax, 9, 4, 10, 4)
    draw_box(ax, 10, 3.5, 2, 1, "Conv\nBlocks", '#ADD8E6')
    
    plt.title("EncoderBlock with ePURE")
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, 'encoder_block.png'), dpi=300)
    plt.close()

def draw_epure():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    
    y_center = 2
    
    # Input
    cx, cy = 0.5, y_center
    draw_box(ax, cx, cy-0.5, 1.5, 1, "Input", '#E0E0E0')
    
    # Conv1
    draw_arrow(ax, 2, cy, 3, cy)
    draw_box(ax, 3, cy-0.5, 1.5, 1, "Conv1\n(32)", '#ADD8E6')
    
    # Conv2
    draw_arrow(ax, 4.5, cy, 5.5, cy)
    draw_box(ax, 5.5, cy-0.5, 1.5, 1, "Conv2\n(32)", '#ADD8E6')
    
    # Residual
    # Loop from after Conv1 (4.5) to after Conv2 (7)
    # Let's just draw box
    draw_arrow(ax, 7, cy, 8, cy, "+ Add")
    
    # SE Block
    draw_box(ax, 8, cy-0.5, 1.5, 1, "SE Block\n(Attn)", '#FFA07A')
    
    # Conv3
    draw_arrow(ax, 9.5, cy, 10.5, cy)
    draw_box(ax, 10.5, cy-0.5, 1.5, 1, "Conv3", '#ADD8E6')
    
    # Final
    draw_arrow(ax, 12, cy, 13, cy)
    draw_box(ax, 13, cy-0.5, 1, 1, "1x1\nOut", '#FFD700')
    
    plt.title("ePURE Module Detail")
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, 'epure_module.png'), dpi=300)
    plt.close()

def draw_maxwell():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)
    
    # Input
    draw_box(ax, 3, 5, 2, 1, "Concat\nFeatures", '#E0E0E0')
    
    # Encoder
    draw_arrow(ax, 4, 5, 4, 4)
    draw_box(ax, 3, 3, 2, 1, "Conv Layers\n(Hidden 32)", '#ADD8E6')
    
    # Output 2 channels
    draw_arrow(ax, 4, 3, 4, 2)
    draw_box(ax, 3, 1.5, 2, 0.5, "2 Channels", '#D3D3D3')
    
    # Split
    draw_arrow(ax, 4, 1.5, 2.5, 0.5)
    draw_arrow(ax, 4, 1.5, 5.5, 0.5)
    
    draw_box(ax, 1.5, 0, 2, 0.5, "Epsilon (ε)", '#90EE90')
    draw_box(ax, 4.5, 0, 2, 0.5, "Sigma (σ)", '#FFB6C1')
    
    plt.title("Maxwell Solver Module")
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, 'maxwell_solver.png'), dpi=300)
    plt.close()

def draw_aspp():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    
    # Input
    draw_box(ax, 4, 7, 2, 0.8, "Input", '#E0E0E0')
    
    # Branches
    y_branch = 5
    x_positions = [1, 3, 5, 7, 9]
    labels = ["1x1 Conv", "3x3 d=3", "3x3 d=6", "3x3 d=9", "Pool+1x1"]
    
    for i, x in enumerate(x_positions):
        draw_arrow(ax, 5, 7, x, 5.5)
        draw_box(ax, x-0.8, y_branch, 1.6, 0.8, labels[i], '#ADD8E6')
        draw_arrow(ax, x, y_branch, 5, 3) 
        
    # Concat
    draw_box(ax, 4, 2, 2, 0.8, "Concat", '#FFE4B5')
    
    # Project
    draw_arrow(ax, 5, 2, 5, 1)
    draw_box(ax, 4, 0.2, 2, 0.8, "Projection", '#FFD700')
    
    plt.title("ASPP Bottleneck")
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, 'aspp_bottleneck.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    draw_encoder_block()
    draw_epure()
    draw_maxwell()
    draw_aspp()
    print("All component diagrams drawn and saved to ./walkthrough/")
