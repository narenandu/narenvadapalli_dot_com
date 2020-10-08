import React from "react";
import { TwitterIcon } from "react-share";

export default function (props) {
    return (
        <div>
            <a href="https://twitter.com/share?ref_src=twsrc%5Etfw" class="twitter-share-button" data-size="large" data-text="Content by @narenandu " data-url="https://www.narenvadapalli.com" data-via="narenandu" data-hashtags="narenandu" data-show-count="false"><TwitterIcon size={64} round={true} /></a><script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
        </div >
    )
}
