test_case = int(input())

for i in range(test_case):
    print("Case " + str(i + 1) + ":")
    
    n_q = input().split(" ")

    n = n_q[0]
    q = n_q[1]

    hard_arr = []
    hard_set_arr = []
    arr = []
    
    for j in range(int(n)):
        my_dict = dict()
        my_set = set()
        str_input = input()

        first_index = 0

        tmp_arr = str_input.split(" ")

        hard_arr.append(str_input)

        for k in range(len(tmp_arr)):
            my_set.add(tmp_arr[k])
            if (tmp_arr[k] in my_dict):
                my_dict[tmp_arr[k]].append(first_index)
            else:
                my_dict[tmp_arr[k]] = [first_index]

            first_index = first_index + len(tmp_arr[k]) + 1

        arr.append(my_dict)
        hard_set_arr.append(my_set)

        
    for k in range(int(q)):
        wanted = input()
        counter = 0

        first_text = ""

        for char in wanted:
            if (char == " "):
                break
            
            first_text += char

        for m in range(int(n)):
            if (first_text in hard_set_arr[m]):
                real_arr = arr[m][first_text]
                text_len = len(wanted)
                
                for index in real_arr:
                    if (wanted == hard_arr[m][index:index+text_len]):
                        counter += 1
                        break

        print(counter)    

