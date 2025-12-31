# leverage_and_kelly
Trying to figure out how much leverage to use in my IBKR account if I really believe the kelly criterion.

# Methodology:
1) Download returns from https://www.investing.com/indices/us-spx-500-historical-data?utm_source=chatgpt.com
   - I don't think this is the best way to do this, but it's the only way I can find that is free and easy to use.
   issues:
   - I don't think this accounts for dividends. Going to arbitarily add _ bps every day to counter this.
   - I don't think this accounts for splits, but splits shouldn't matter in SPX
   - Expense ratio is sooo small, but I'll take out VOO's 3bps a year. 
2) Pull overnight treasury returns from ___.
3) Assume that I pay an extra 1% a year on every dollar of margin.
4) Calculate my returns starting 1950 using real data.

# optional:: 

5) Account for the fact that I don't rebalance very often.
6) Bootstrap to see if affecting order of returns affects things much
    - vol is autocorrleated, which is annoying, so maybe when I bootstrap I pull a random 2-10 day period with repalcement. Sounds fine to me. Even better would be to have vol be a random walk or something, but I'm so likely to mess that up. 


# div readjusting:
Mid-1970s to early-1980s: commonly ~4%–5% 
Multpl

Late-1980s: often ~3%–4% 
Multpl

1990s: drifted down to roughly ~1%–3% (lower by late 1990s) 
Multpl

2000s–2010s: mostly around ~1.5%–2.5% 
Multpl

Mid-2020s (recent): about ~1%–1.5% (currently around ~1.15%)