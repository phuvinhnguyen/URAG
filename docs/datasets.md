# Normal result full

\begin{table*}[t]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \caption{\textbf{Accuracy, Coverage, and Uncertainty results of different RAG methods across tasks using Llama-3.1-8B-Instruct.} \textbf{``W/o Retrieve''} denotes the baseline without retrieval. For each dataset, the top three methods in terms of highest accuracy and lowest uncertainty are highlighted in red.}
  \label{tab:rag_unc_llm_8b}
  \begin{adjustbox}{center, max width=\textwidth}
  \begin{tabularx}{.85\textwidth}{l *{8}{c}}
    \toprule
    \multicolumn{1}{l}{} & 
    \multicolumn{1}{c}{\textbf{Healthcare}} & 
    \multicolumn{2}{c}{\textbf{Code}} & 
    \multicolumn{1}{c}{\textbf{Research}} & 
    \multicolumn{1}{c}{\textbf{Math}} &
    \multicolumn{3}{c}{\textbf{General Text}} \\
    \cmidrule(lr){2-2}\cmidrule(lr){3-4}\cmidrule(lr){5-5}\cmidrule(lr){6-6}\cmidrule(lr){7-9}
    \textbf{RAG} &
    \textbf{Healthver} &
    \textbf{ODEX} &
    \textbf{LCA} &
    \textbf{SciFact} &
    \textbf{Olympiad} &
    \textbf{CRAG} &
    \textbf{NewsSum} &
    \textbf{DialFact} \\
    \midrule

    \multicolumn{9}{c}{\textit{Performance} -- \textbf{Acc (\%) $\uparrow$}} \\
    \cdashline{1-9}[2.5pt/5pt]\noalign{\vskip 0.5ex}
\textbf{W/o Retrieve} & 0.45 & 0.88 & 0.21 & 0.45 & 0.34 & 0.55 & 0.36 & 0.47 \\
\textbf{FiD} & 0.37 & 0.28 & 0.21 & 0.42 & 0.30 & 0.31 & 0.26 & 0.35 \\
\textbf{Fusion} & \textcolor{red}{0.52} & \textcolor{red}{0.86} & \textcolor{red}{0.82} & 0.69 & 0.37 & 0.66 & \textcolor{red}{0.41} & 0.71 \\
\textbf{HyDE} & \textcolor{red}{0.53} & 0.85 & 0.73 & \textcolor{red}{0.72} & 0.40 & 0.62 & \textcolor{red}{0.43} & \textcolor{red}{0.72} \\
\textbf{RAPTOR} & 0.51 & 0.85 & 0.73 & 0.70 & 0.39 & 0.67 & 0.38 & \textcolor{red}{0.72} \\
\textbf{RAT} & 0.49 & 0.84 & 0.34 & 0.65 & \textcolor{red}{0.46} & 0.67 & \textcolor{red}{0.40} & 0.64 \\
\textbf{REPLUG} & 0.51 & \textcolor{red}{0.86} & 0.73 & 0.70 & 0.36 & 0.67 & 0.38 & 0.71 \\
\textbf{Self-RAG} & 0.51 & 0.83 & \textcolor{red}{0.74} & 0.70 & 0.40 & 0.63 & 0.38 & 0.68 \\
\textbf{Naive} & \textcolor{red}{0.54} & \textcolor{red}{0.86} & \textcolor{red}{0.76} & 0.70 & 0.40 & \textcolor{red}{0.68} & 0.37 & \textcolor{red}{0.72} \\
    \midrule

    \multicolumn{9}{c}{\textit{Coverage Rate} -- \textbf{CR (\%) $\uparrow$}} \\
    \cdashline{1-9}[2.5pt/5pt]\noalign{\vskip 0.5ex}
\textbf{W/o Retrieve} & 0.90 & 0.92 & 0.91 & 0.90 & 0.89 & 0.90 & 0.87 & 0.93 \\
\textbf{FiD} & 1.00 & 0.95 & 0.94 & 1.00 & 0.95 & 0.94 & 0.99 & 0.94 \\
\textbf{Fusion} & 0.91 & 0.94 & 0.92 & 0.90 & 0.87 & 0.92 & 0.91 & 0.91 \\
\textbf{HyDE} & 0.91 & 0.93 & 0.91 & 0.89 & 0.93 & 0.91 & 0.89 & 0.90 \\
\textbf{RAPTOR} & 0.92 & 0.93 & 0.94 & 0.88 & 0.88 & 0.91 & 0.90 & 0.91 \\
\textbf{RAT} & 0.90 & 0.92 & 0.93 & 0.92 & 0.87 & 0.91 & 0.90 & 0.90 \\
\textbf{REPLUG} & 0.93 & 0.97 & 0.97 & 0.90 & 0.89 & 0.90 & 0.92 & 0.95 \\
\textbf{Self} & 0.90 & 0.92 & 0.94 & 0.93 & 0.90 & 0.92 & 0.87 & 0.91 \\
\textbf{Naive} & 0.92 & 0.92 & 0.93 & 0.86 & 0.91 & 0.92 & 0.90 & 0.90 \\
    \midrule

    \multicolumn{9}{c}{\textit{Prediction Uncertainty} -- \textbf{SS $\downarrow$}} \\
    \cdashline{1-9}[2.5pt/5pt]\noalign{\vskip 0.5ex}
\textbf{W/o Retrieve} & 2.62 & \textcolor{red}{1.66} & 4.64 & 2.59 & 3.48 & 2.61 & 2.94 & 2.55 \\
\textbf{FiD} & 3.00 & 3.63 & 4.76 & 3.00 & 3.98 & 3.69 & 3.98 & 2.84 \\
\textbf{Fusion} & 2.64 & 1.73 & \textcolor{red}{2.19} & 2.18 & 3.36 & 2.38 & 2.82 & 2.00 \\
\textbf{HyDE} & 2.49 & 1.68 & 2.38 & 2.00 & 3.69 & 2.42 & 2.74 & \textcolor{red}{1.97} \\
\textbf{RAPTOR} & 2.65 & 1.71 & 2.70 & 1.98 & 3.40 & \textcolor{red}{2.32} & 2.69 & 2.05 \\
\textbf{RAT} & 2.58 & 1.70 & 4.47 & 2.46 & \textcolor{red}{3.30} & 2.50 & 3.05 & 2.22 \\
\textbf{REPLUG} & \textcolor{red}{2.24} & 3.50 & 4.63 & 2.57 & 3.85 & 3.72 & 3.73 & 2.69 \\
\textbf{Self} & 2.69 & 1.77 & 2.59 & 2.32 & 3.51 & 2.52 & \textcolor{red}{2.66} & 2.05 \\
\textbf{Naive} & 2.65 & 1.69 & 2.46 & \textcolor{red}{1.94} & 3.54 & 2.29 & 2.77 & 2.05 \\
    \bottomrule
      \end{tabularx}
  \end{adjustbox}
\end{table*}

# Self-aware result full

\begin{table*}[t]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \caption{\textbf{Accuracy, Coverage, and uncertainty results of different RAG methods across tasks using 8B LLM in Self-Aware Evaluation Setting (models are provided with their own confidence scores)}. The number indicates the uncertainty of the whole RAG, while the subscripts indicate the uncertainty calibrated by LLM uncertainty. This is for showing the effect of contexts on LLM.}
  \label{tab:rag_unc_llm_8b_self-aware}
  \begin{adjustbox}{center, max width=\textwidth}
  \begin{tabularx}{\textwidth}{l *{10}{c}}
    \toprule
    \multicolumn{1}{l}{} & 
    \multicolumn{1}{c}{\textbf{Healthcare}} & 
    \multicolumn{2}{c}{\textbf{Code}} & 
    \multicolumn{1}{c}{\textbf{Research}} & 
    \multicolumn{1}{c}{\textbf{Math}} &
    \multicolumn{3}{c}{\textbf{General Text}} &
    \multicolumn{2}{c}{\textbf{Irrelevant Contexts}} \\
    \cmidrule(lr){2-2}\cmidrule(lr){3-4}\cmidrule(lr){5-5}\cmidrule(lr){6-6}\cmidrule(lr){7-9}\cmidrule(lr){10-11}
    \textbf{RAG} &
    \textbf{Healthver} &
    \textbf{Odex} &
    \textbf{LCA} &
    \textbf{SciFact} &
    \textbf{Olympiad} &
    \textbf{CRAG} &
    \textbf{NewsSum} &
    \textbf{DialFact} &
    \textbf{W/DialFact} &
    \textbf{W/Odex} \\
    \midrule

    \multicolumn{11}{c}{\textit{Performance} -- \textbf{Acc (\%) $\uparrow$}} \\
\cdashline{1-11}[2.5pt/5pt]\noalign{\vskip 0.5ex}
\textbf{W/o Retrieve} & 0.45 & 0.88 & 0.21 & 0.45 & 0.37 & 0.60 & 0.37 & 0.46 & 0.43 & 0.87 \\
\textbf{FiD} & 0.36 & 0.27 & 0.22 & 0.43 & 0.30 & 0.32 & 0.24 & 0.35 & 0.33 & 0.26 \\
\textbf{Fusion} & 0.52 & 0.84 & 0.77 & 0.70 & 0.38 & 0.68 & 0.39 & 0.70 & 0.34 & 0.87 \\
\textbf{HyDE} & 0.51 & 0.85 & 0.74 & 0.72 & 0.37 & 0.67 & 0.42 & 0.71 & 0.31 & 0.88 \\
\textbf{RAPTOR} & 0.51 & 0.85 & 0.73 & 0.69 & 0.38 & 0.67 & 0.38 & 0.71 & 0.34 & 0.87 \\
\textbf{RAT} & 0.51 & 0.68 & 0.28 & 0.44 & 0.39 & 0.48 & 0.39 & 0.44 & 0.37 & 0.77 \\
\textbf{REPLUG} & 0.51 & 0.86 & 0.73 & 0.70 & 0.38 & 0.67 & 0.38 & 0.71 & 0.34 & 0.87 \\
\textbf{Self-RAG} & 0.51 & 0.83 & 0.74 & 0.70 & 0.41 & 0.64 & 0.38 & 0.67 & 0.38 & 0.86 \\
\textbf{Naive} & 0.51 & 0.86 & 0.73 & 0.69 & 0.39 & 0.67 & 0.38 & 0.71 & 0.34 & 0.87 \\
\midrule

\multicolumn{11}{c}{\textit{Coverage Rate} -- \textbf{CR (\%) $\uparrow$}} \\
\cdashline{1-11}[2.5pt/5pt]\noalign{\vskip 0.5ex}
\textbf{W/o Retrieve} & 0.92 & 0.92 & 0.91 & 0.90 & 0.90 & 0.90 & 0.90 & 0.88 & 0.92 & 0.92 \\
\textbf{FiD} & 0.94 & 0.96 & 0.98 & 0.96 & 0.95 & 0.94 & 0.97 & 0.94 & 0.93 & 0.96 \\
\textbf{Fusion} & 0.90 & 0.90 & 0.89 & 0.89 & 0.90 & 0.89 & 0.92 & 0.89 & 0.91 & 0.93 \\
\textbf{HyDE} & 0.88 & 0.87 & 0.88 & 0.87 & 0.89 & 0.88 & 0.90 & 0.90 & 0.92 & 0.93 \\
\textbf{RAPTOR} & 0.91 & 0.89 & 0.88 & 0.92 & 0.91 & 0.90 & 0.89 & 0.90 & 0.92 & 0.92 \\
\textbf{RAT} & 0.91 & 0.92 & 0.93 & 0.88 & 0.92 & 0.89 & 0.89 & 0.88 & 0.92 & 0.92 \\
\textbf{REPLUG} & 0.93 & 0.97 & 0.97 & 0.90 & 0.88 & 0.90 & 0.92 & 0.95 & 0.96 & 0.98 \\
\textbf{Self} & 0.90 & 0.92 & 0.88 & 0.89 & 0.89 & 0.89 & 0.89 & 0.89 & 0.91 & 0.92 \\
\textbf{Naive} & 0.91 & 0.89 & 0.90 & 0.90 & 0.92 & 0.89 & 0.90 & 0.89 & 0.92 & 0.92 \\
\midrule

\multicolumn{11}{c}{\textit{Prediction Uncertainty} -- \textbf{SS $\downarrow$}} \\
\cdashline{1-11}[2.5pt/5pt]\noalign{\vskip 0.5ex}
\textbf{W/o Retrieve} & 2.72 & 1.65 & 4.68 & 2.67 & 3.76 & 2.98 & 3.48 & 2.59 & 2.55 & 1.52 \\
\textbf{FiD} & 2.85 & 3.77 & 4.82 & 2.85 & 3.98 & 3.69 & 3.85 & 2.83 & 2.84 & 3.77 \\
\textbf{Fusion} & 2.62 & 1.51 & 3.77 & 2.40 & 3.65 & 2.90 & 3.40 & 2.35 & 2.59 & 1.66 \\
\textbf{HyDE} & 2.34 & 1.64 & 4.30 & 2.39 & 3.73 & 3.08 & 3.58 & 2.44 & 2.63 & 1.67 \\
\textbf{RAPTOR} & 2.68 & 1.51 & 3.84 & 2.43 & 3.70 & 2.98 & 3.24 & 2.44 & 2.69 & 1.55 \\
\textbf{RAT} & 2.79 & 2.07 & 4.57 & 2.62 & 3.74 & 2.85 & 3.41 & 2.55 & 2.71 & 1.84 \\
\textbf{REPLUG} & 2.24 & 3.50 & 4.61 & 2.57 & 3.83 & 3.66 & 3.73 & 2.69 & 2.80 & 3.50 \\
\textbf{Self} & 2.50 & 1.56 & 4.07 & 2.48 & 3.67 & 2.95 & 3.35 & 2.50 & 2.63 & 1.59 \\
\textbf{Naive} & 2.64 & 1.48 & 3.79 & 2.34 & 3.74 & 2.91 & 3.21 & 2.40 & 2.69 & 1.55 \\
    \bottomrule
  \end{tabularx}
  \end{adjustbox}
\end{table*}


# Irrelevant context

\begin{table*}[t]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{3.5pt}
  \caption{\textbf{Accuracy, Coverage, and Uncertainty results of different RAG methods on three datasets.} The retrieval system is poisoned and returns irrelevant documents. Subscripts show differences from normal context (irrelevant - normal).}
  \label{tab:rag_results_reorganized}
  \begin{adjustbox}{center, max width=0.95\textwidth}
  \begin{tabular}{lccccccccc}
    \toprule
    & \multicolumn{9}{c}{\textbf{Model}} \\
    \cmidrule(l){2-10}
    \textbf{Dataset} & 
    \rotatebox[origin=c]{0}{\textbf{W/o}} & 
    \textbf{Fid} & 
    \textbf{Fusion} & 
    \textbf{HyDE} & 
    \textbf{RAPTOR} & 
    \textbf{RAT} & 
    \textbf{REPLUG} & 
    \textbf{Self} & 
    \textbf{Naive} \\
    \cmidrule(lr){1-1}\cmidrule(l){2-10}
    \textbf{Odex} \\
    \quad Acc (\%) $\uparrow$ & 0.87$_{-0.01}$ & 0.27$_{-0.01}$ & 0.84$_{-0.02}$ & 0.86$_{+0.01}$ & 0.85$_{+0.00}$ & 0.85$_{+0.01}$ & 0.85$_{-0.01}$ & 0.85$_{+0.02}$ & 0.85$_{-0.01}$ \\
    \quad CR (\%) $\uparrow$   & 0.92$_{+0.00}$ & 0.94$_{-0.01}$ & 0.93$_{-0.01}$ & 0.92$_{-0.01}$ & 0.93$_{+0.00}$ & 0.92$_{+0.00}$ & 0.97$_{+0.00}$ & 0.93$_{+0.01}$ & 0.93$_{+0.01}$ \\
    \quad SS $\downarrow$     & 1.66$_{+0.00}$ & 3.77$_{+0.14}$ & 1.67$_{-0.06}$ & 1.70$_{+0.02}$ & 1.64$_{-0.07}$ & 1.69$_{-0.01}$ & 3.50$_{+0.00}$ & 1.72$_{-0.05}$ & 1.64$_{-0.05}$ \\\hline
    \addlinespace[0.3em]
    \textbf{LCA} \\
    \quad Acc (\%) $\uparrow$ & 0.21$_{+0.00}$ & 0.18$_{-0.03}$ & 0.22$_{-0.60}$ & 0.29$_{-0.44}$ & 0.24$_{-0.49}$ & 0.18$_{-0.16}$ & 0.23$_{-0.50}$ & 0.20$_{-0.54}$ & 0.22$_{-0.54}$ \\
    \quad CR (\%) $\uparrow$   & 0.91$_{+0.00}$ & 0.97$_{+0.03}$ & 0.91$_{-0.01}$ & 0.84$_{-0.07}$ & 0.92$_{-0.02}$ & 0.95$_{+0.02}$ & 0.97$_{+0.00}$ & 0.93$_{-0.01}$ & 0.92$_{-0.01}$ \\
    \quad SS $\downarrow$     & 4.64$_{+0.00}$ & 4.82$_{+0.06}$ & 4.45$_{+2.26}$ & 3.90$_{+1.52}$ & 4.47$_{+1.77}$ & 4.64$_{+0.17}$ & 4.91$_{+0.28}$ & 4.55$_{+1.96}$ & 4.47$_{+2.01}$ \\\hline
    \addlinespace[0.3em]
    \textbf{DialFact} \\
    \quad Acc (\%) $\uparrow$ & 0.46$_{-0.01}$ & 0.34$_{-0.01}$ & 0.47$_{-0.24}$ & 0.44$_{-0.28}$ & 0.47$_{-0.25}$ & 0.52$_{-0.12}$ & 0.47$_{-0.24}$ & 0.39$_{-0.29}$ & 0.47$_{-0.25}$ \\
    \quad CR (\%) $\uparrow$   & 0.93$_{+0.00}$ & 0.94$_{+0.00}$ & 0.89$_{-0.02}$ & 0.90$_{+0.00}$ & 0.90$_{-0.01}$ & 0.90$_{+0.00}$ & 0.96$_{+0.01}$ & 0.89$_{-0.02}$ & 0.90$_{+0.00}$ \\
    \quad SS $\downarrow$     & 2.55$_{+0.00}$ & 2.83$_{-0.01}$ & 2.33$_{+0.33}$ & 2.42$_{+0.45}$ & 2.36$_{+0.31}$ & 2.49$_{+0.27}$ & 2.75$_{+0.06}$ & 2.48$_{+0.43}$ & 2.36$_{+0.31}$ \\
    \bottomrule
  \end{tabular}
  \end{adjustbox}
\end{table*}