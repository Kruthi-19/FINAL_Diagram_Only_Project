import random, numpy as np

def generate_candidate_layouts(grid, event_data, count=5):
    stall_count=max(1,min(int(event_data.get("stalls",8)),20))
    rows,cols=grid.shape
    patterns=["left_right_rows","center_split","side_only","u_shape","distributed"]
    return [{"name":f"Layout {i+1} - {patterns[i].replace('_',' ').title()}","pattern":patterns[i],"stalls":create_stalls(patterns[i],stall_count,rows,cols)} for i in range(count)]

def create_stalls(pattern, n, rows, cols):

    stalls = []

    cur = 0

    start_y = int(rows * 0.62)

    if pattern == "left_right_rows":

        for r in range((n + 1) // 2):

            y = start_y + r * 3

            if cur < n:

                stalls.append((
                    min(y, rows - 5),
                    int(cols * .22)
                ))

                cur += 1

            if cur < n:

                stalls.append((
                    min(y, rows - 5),
                    int(cols * .72)
                ))

                cur += 1


    elif pattern == "center_split":

        for r in range((n + 1) // 2):

            y = start_y + r * 3

            if cur < n:

                stalls.append((
                    min(y, rows - 5),
                    int(cols * .35)
                ))

                cur += 1

            if cur < n:

                stalls.append((
                    min(y, rows - 5),
                    int(cols * .60)
                ))

                cur += 1

  

    elif pattern == "side_only":

        for r in range(n):

            x = int(cols * .15) if r % 2 == 0 else int(cols * .82)

            y = start_y + (r // 2) * 3

            stalls.append((
                min(y, rows - 5),
                x
            ))


    elif pattern == "u_shape":

        for i in range(n):

            if i < n // 3:

                stalls.append((
                    start_y,
                    min(cols - 5, int(cols * .25) + i * 4)
                ))

            elif i < 2 * n // 3:

                stalls.append((
                    start_y + 4,
                    int(cols * .22)
                ))

            else:

                stalls.append((
                    start_y + 4,
                    int(cols * .75)
                ))



    else:

        for i in range(n):

            y = random.randint(start_y, rows - 5)

            x = random.choice([
                random.randint(5, 14),
                random.randint(cols - 15, cols - 6)
            ])

            stalls.append((y, x))

    return stalls

def score_layout(layout,grid,zone_map,density):
    exits=[z["center"] for z in zone_map if z["label"]=="exit"]
    stages=[z["center"] for z in zone_map if z["label"]=="stage"]
    penalty=0; reward=0
    for y,x in layout["stalls"]:
        y1,y2=max(0,y-1),min(grid.shape[0],y+2); x1,x2=max(0,x-1),min(grid.shape[1],x+2)
        if np.any(grid[y1:y2,x1:x2]==1): penalty+=30
        for ex in exits:
            d=abs(y-ex[0])+abs(x-ex[1])
            penalty += 20 if d<8 else 0
            reward += min(d,20)*.2
        for st in stages:
            d=abs(y-st[0])+abs(x-st[1])
            if d<7: penalty+=15
        penalty += density[y,x]*4
    stalls=layout["stalls"]; spread=0
    for i in range(len(stalls)):
        for j in range(i+1,len(stalls)):
            spread += min(abs(stalls[i][0]-stalls[j][0])+abs(stalls[i][1]-stalls[j][1]),10)
    reward += spread*.15
    fitness=max(0,min(100,100+reward-penalty))
    layout["fitness"]=round(float(fitness),2)
    layout["safety"]=round(layout["fitness"]/10,1)
    layout["congestion"]=round(max(0,10-layout["safety"]),1)
    layout["accessibility"]=round(min(10,5+layout["fitness"]/20),1)
    layout["circulation"]=round(min(10,4+layout["fitness"]/18),1)
    return layout

def optimize_layouts(grid,zone_map,density,event_data):
    layouts=generate_candidate_layouts(grid,event_data)
    scored=[score_layout(l,grid,zone_map,density) for l in layouts]
    best=max(scored,key=lambda x:x["fitness"])
    return scored,best
