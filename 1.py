def add(i, label):
    label.append(i)
    return label


def test(label):
    for i in range(8):
        label = add(i)
    return label


label = []
label = test(label)
print(label)
