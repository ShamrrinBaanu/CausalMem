import networkx as nx
import plotly.graph_objects as go
from causal_graph import build_graph

def graph_figure(user_id):
    graph=build_graph(user_id); pos=nx.spring_layout(graph,seed=17) if graph.number_of_nodes() else {}
    ex=[];ey=[];nxp=[];nyp=[];labels=[]
    for a,b in graph.edges(): ex.extend([pos[a][0],pos[b][0],None]); ey.extend([pos[a][1],pos[b][1],None])
    for n,d in graph.nodes(data=True): nxp.append(pos[n][0]);nyp.append(pos[n][1]);labels.append(d.get("state_value",""))
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=ex,y=ey,mode="lines",line=dict(width=1,color="#64748b"),hoverinfo="none"))
    fig.add_trace(go.Scatter(x=nxp,y=nyp,mode="markers+text",text=labels,textposition="bottom center",marker=dict(size=22,color="#38bdf8"),hovertext=labels,hoverinfo="text"))
    fig.update_layout(showlegend=False,height=360,margin=dict(l=10,r=10,t=10,b=10),xaxis=dict(visible=False),yaxis=dict(visible=False))
    return fig
