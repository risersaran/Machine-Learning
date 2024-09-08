import pandas as pd

def generalize_S(s, h):
    return ['?' if s[i] != h[i] else s[i] for i in range(len(s))]

def specialize_G(g, h, domains):
    result = []
    for i in range(len(g)):
        if g[i] == '?':
            for val in domains[i]:
                if val != h[i]:
                    g_new = g.copy()
                    g_new[i] = val
                    result.append(g_new)
    return result

def candidate_elimination(examples):
    domains = []
    for attr in examples.columns[:-1]:
        domains.append(list(examples[attr].unique()))
    
    G = [['?' for _ in range(len(examples.columns) - 1)]]
    S = ['0' for _ in range(len(examples.columns) - 1)]
    
    for _, example in examples.iterrows():
        example = example.tolist()
        if example[-1] == 'Yes':
            G = [g for g in G if all(g[i] == '?' or g[i] == example[i] for i in range(len(example) - 1))]
            S = generalize_S(S, example)
        elif example[-1] == 'No':
            S = S
            G = [g for g in G if g != ['?' for _ in range(len(example) - 1)]]
            for g in G.copy():
                if all(g[i] == '?' or g[i] == example[i] for i in range(len(example) - 1)):
                    G.remove(g)
                    G.extend(specialize_G(g, example, domains))
        
        G = [g for g in G if any(all(s != '0' and (s == '?' or s == g[i]) for i, s in enumerate(S)))]
        S = [s if s != '0' and any(g[i] == '?' or g[i] == s for g in G) else '0' for i, s in enumerate(S)]
        
        print(f"For Example {example}:")
        print("G:", G)
        print("S:", S)
        print()
    
    return G, S

def main():
    file_path = r'C:\Users\Welcome\OneDrive\Documents\training.csv'
    training_data = pd.read_csv(file_path)
    
    G, S = candidate_elimination(training_data)
    
    print("Final Version Space:")
    print("G:", G)
    print("S:", S)

if __name__ == "__main__":
    main()
