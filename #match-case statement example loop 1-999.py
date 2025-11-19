#match-case statement example loop 1-99999
for i in range(1, 100000):
    match str(i).length:
        
        case 1:
            print(f"0000{i}")
        case 2:
            print(f"000{i}")
        case 3:
            print(f"00{i}")
        case 4:
            print(f"0{i}")
