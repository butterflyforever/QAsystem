Format of Requests and Responses

[{
    'msgType': msgType,
    'content': content,
    'userId': userId
}, {
    'msgType': msgType,
    'content': content,
    'userId': userId
} ...]

msgType: 'none', 'text', 'image', 'new', 'other'
content: list of words (already segmented), e.g. ['今天', '天气', '怎么样', '？']