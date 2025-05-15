from graphviz import Digraph

dot = Digraph(comment='Lydfilter Flow')

dot.node('A', 'Optagelse')
dot.node('B', 'Low-pass Filter (200/500/1000/2000 Hz)')
dot.node('C', 'Afspilning')
dot.node('D', 'Gem WAV-fil')
dot.node('E', 'Plot (Tidsdomæne og evt. Frekvensdomæne)')

dot.edges(['AB', 'BC', 'BD', 'BE'])

dot.render('flowdiagram', format='png', view=True)
