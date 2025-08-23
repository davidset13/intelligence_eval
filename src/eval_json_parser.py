import ast


# HLE, MMLU, GPQA, and all of LiveBench except for Instruction Following
def parse_eval_json(response: str) -> bool:
    try:
        correct = ast.literal_eval(response)['correct']
        if correct == 'yes':
            return True
        else:
            return False
    except:
        try:
            if ('"correct":' in response or "'correct':" in response):
                idx1 = response.find('"correct":')
                idx2 = response.find("'correct':")
                if idx1 != -1 and idx2 != -1:
                    return False
                elif idx1 != -1:
                    if_yes = response[idx1:].find('yes')
                    if_no = response[idx1:].find('no')
                    if if_yes == -1:
                        return False
                    elif if_yes < if_no:
                        return True
                    else:
                        return False
                elif idx2 != -1:
                    if_yes = response[idx2:].find('yes')
                    if_no = response[idx2:].find('no')
                    if if_yes == -1:
                        return False
                    elif if_yes < if_no:
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                return False
        except:
            return False


# Instruction Following LiveBench
def eval_livebench_if(response: str) -> float:
    try:
        response_dict = ast.literal_eval(response)
        instructions_followed = response_dict['instructions_followed']
        correctness = response_dict['correctness']
        if instructions_followed == 'yes' and correctness == 'yes':
            return 1
        elif instructions_followed == 'yes' or correctness == 'yes':
            return 0.5
        else:
            return 0
    except:
        parsed_resp = 0
        try:
            if ('"instructions_followed":' in response or "'instructions_followed':" in response):
                idx1 = response.find('"instructions_followed":')
                idx2 = response.find("'instructions_followed':")
                if idx1 != -1 and idx2 != -1:
                    pass
                elif idx1 != -1:
                    if_yes = response[idx1:].find('yes')
                    if_no = response[idx1:].find('no')
                    if if_yes == -1:
                        pass
                    elif if_yes < if_no:
                        parsed_resp = 0.5
                elif idx2 != -1:
                    if_yes = response[idx2:].find('yes')
                    if_no = response[idx2:].find('no')
                    if if_yes == -1:
                        pass
                    elif if_yes < if_no:
                        parsed_resp = 0.5
        except:
            pass
        
        try:
            if ('"correctness":' in response or "'correctness':" in response):
                idx1 = response.find('"correctness":')
                idx2 = response.find("'correctness':")
                if idx1 != -1 and idx2 != -1:
                    pass
                elif idx1 != -1:
                    if_yes = response[idx1:].find('yes')
                    if_no = response[idx1:].find('no')
                    if if_yes == -1:
                        pass
                    elif if_yes < if_no:
                        parsed_resp += 0.5
                elif idx2 != -1:
                    if_yes = response[idx2:].find('yes')
                    if_no = response[idx2:].find('no')
                    if if_yes == -1:
                        pass
                    elif if_yes < if_no:
                        parsed_resp += 0.5
        except:
            pass

        return parsed_resp