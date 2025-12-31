# leverage_and_kelly
Trying to figure out how much leverage to use in my IBKR account if I really believe the kelly criterion.

# Methodology:
1) Download returns from https://www.investing.com/indices/us-spx-500-historical-data?utm_source=chatgpt.com
   - I don't think this is the best way to do this, but it's the only way I can find that is free and easy to use.
   issues:
   - I don't think this accounts for dividends. Going to arbitarily add a fraction of a bp every day to counter this, see note below.
   - No splits to worry about in SPX.
   - Expense ratio of VOO in the modern era is 3bps/year, I'll take this out on a daily basis. 
2) Pull fed funds rate from the fred website, lost exact URL. This is monthly data, I just forward filled. 
    Assume that I pay an extra 1% a year on every dollar of margin.
4) Calculate my returns starting 1980 using real data.
5) Bootstrap to see if affecting order of returns affects things much
    - vol is autocorrleated, which is annoying, so maybe when I bootstrap I pull a random 3-10 day period with replacement. Even better would be to have vol be a random walk and use some fractal/mandelbrot/lognormal distribution or something, but I'm so likely to mess that up and can't be asked.
6) I think I handle subtracting RFR on leverage properly. I charge myself three days of RFR on Mondays. 

## div readjusting
Underlyings of SPX pay divs, and the index does not adjust for this. I'm just going to add some tiny amount to each
day's returns to account for this. 
Using the following schedule, pulled straight from chatGPT, didn't check.
div_schedule = {
    (1975, 1982): 0.045,  # 4.5% mid-70s to early-80s
    (1983, 1989): 0.035,  # 3.5% late-80s
    (1990, 1999): 0.025,  # 2.5% 1990s
    (2000, 2019): 0.020,  # 2.0% 2000s-2010s
    (2020, 2025): 0.0125, # 1.25% mid-2020s
}

# Conclusions:
This code suggests that kelly optimal leverage is 2x. That's suspiciously high and my balls aren't that big. In real acadmic papers (I've only read one or two and asked Chat to summarize the rest) they suggest something like 1.5x. Anyway, I have convinced myself that no one is making crazy assumptions when they suggest 1.5x, and I should nut up and increase my leverage (it's something like 1.2x rn in my brokerage account, much lower if you account for the fact that I have a lot of net worth tied up in 401ks). 

# Further work:
1) Account for the fact that I don't rebalance daily. I saw some paper that said that this took optimal leverage from 1.5x to 1.4x or something. I tell myself that I shouldn't rebalace (or at least that it doesn't matter) because I'm happier to be long the market when it's in turmoil, but I think all empirical evidence points to the contrary. 
2) Do a better job adjusting for dividends. 
3) Model the fact that I have money tied up in places I can't use leverage, ie my 401k.